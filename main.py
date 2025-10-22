import pandas as pd
import json
import whisper 
from openai import OpenAI
import warnings
import soundfile as sf
import sounddevice as sd
import numpy as np
import wave
import pyttsx3
import keyboard
import time
import os
from datetime import datetime
warnings.filterwarnings("ignore")
import chromadb
from dotenv import load_dotenv


# create / load persistent database folder
persist_path = os.path.abspath("chroma_db")
chroma_client = chromadb.PersistentClient(path=persist_path)
# one collection to hold all customer memories
memory_collection = chroma_client.get_or_create_collection("customer_memory")

print("After restart, count:", memory_collection.count())



client = OpenAI(api_key="YOUR_API_KEY_HERE")


client = OpenAI(api_key="API_KEY")
sap_data = pd.read_excel("demo_sap.xlsx")
model = whisper.load_model("small")

conversation_log = []

SAMPLE_RATE = 16000  # Whisper prefers 16kHz
CHANNELS = 1
THRESHOLD = 0.02
CHUNK_SIZE = 1024

LOG_DIR = "logs"

def transcribe_audio(file_path: str) -> str:
    print(f"Transcribing {file_path}")
    result = model.transcribe(file_path)
    text = result['text']
    print(f"Transcribed text: {text}")
    return text

def create_new_log_file():
    os.makedirs(LOG_DIR, exist_ok=True)

    existing_logs = [f for f in os.listdir(LOG_DIR) if f.startswith("conversation_log_") and f.endswith(".json")]
    if existing_logs:
        indices = [int(f.split("_")[-1].split(".")[0]) for f in existing_logs if f.split("_")[-1].split(".")[0].isdigit()]
        next_index = max(indices) + 1 if indices else 1
    else:
        next_index = 1

    filename = f"conversation_log_{next_index}.json"
    filepath = os.path.join(LOG_DIR, filename)

    conversation_log = {
        "session_id": next_index,
        "created_at": datetime.now().isoformat(),
        "resolved": False,
        "entries": []
    }

    with open(filepath, 'w') as f:
        json.dump(conversation_log, f, indent=4)

    print(f"Created new log: {filepath}")
    return filepath


def load_conversation_log(file_name):
    with open(file_name, "r") as f:
        return json.load(f)
    
def save_conversation_log(filepath, conversation_log):
    with open(filepath, 'w') as f:
        json.dump(conversation_log, f, indent = 4)


def store_memory(phone_number: int, user_input: str, ai_reply:str):
    #  Save one interaction into Chroma memory for future retrieval
    try:
        text = f"Customer: {user_input},\n AI:{ai_reply}"

        #embedding vector (sematic representaion)
        embedding = client.embeddings.create(
            model= "text-embedding-3-small",
            input= text
        ).data[0].embedding

        memory_collection.add(
            ids=[f"{phone_number}_{hash(text)}"],
            documents=[text],
            metadatas=[{"phone": str(phone_number)}],
            embeddings=[embedding],
        )

        print(f"stored memory for {phone_number}")
        

    except Exception as e:
        print(f"Memory store failed {e}")

def retrieve_memeory(phone_number:int, query:str, n_results: int =3):
    #Fetch the most relevant past exchanges for this phone number.
    try:
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding

        results = memory_collection.query(
            query_embeddings=[embedding],
            n_results= n_results,
            where={"phone": str(phone_number)}
        )

        matches = results.get("documents", [[]])[0]
        if matches:
            print(f"retrieved {len(matches)}, memories from {phone_number}")
        return matches
    except Exception as a:
        print("Retrival failed")
        return []

def text_to_speech(text, voice="alloy"):
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text
    ) as response:
        response.str()



def text_to_speech_fallback(text):
    #fallback if chat gpt not working

    if not text.strip():
        return

    print("speaking locally")
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.setProperty('volume', 1)

    engine.say(text)
    engine.runAndWait()


def play_audio(audio_file):
    data, samplerate = sf.read(audio_file)
    sd.play(data, samplerate)
    sd.wait()


def speech_toggle():
    print("Hold spacebar to talk")
    while not keyboard.is_pressed("space"):
        time.sleep(0.01)
    print("RECORDING")            
 
    frames = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
    with stream:
        while keyboard.is_pressed("space"):
            data, _ = stream.read(1024)
            frames.append(data)
    print("STOPPED RECORDING")

    audio = np.concatenate(frames, axis=0)

    # Convert to int16 and write WAV file
    filename = "input.wav"
    int16 = (audio * 32767).astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(int16.tobytes())

    print(f"Saved recording to {filename}")
    return filename
        


def fetch_sap_info(phone_number: int) -> dict:
    result = sap_data[sap_data['CustomerPhone'] == phone_number]
    if not result.empty:
        return result.to_dict(orient='records')[0]
    return None


def ai_response(user_text: str, sap_info: dict = None, phone_number:int = None) -> str:
    system_prompt = """
    You are a voice-based customer service assistant.
    You can access a history of this customer's past interactions (retrieved from memory).
    Use that history as if you personally recall those past conversations.
    If the customer asks about "our last chat" or "what I said before", summarize or quote from those past interactions.

    Rules:
    - Keep replies under 40 words unless absolutely necessary.
    - Speak conversationally — sound like a real person, not a bot.
    - Do not mention memory retrieval, data storage, or being an AI.
    - Use the customer's SAP data and past interactions naturally in responses.
    - If an issue cannot be solved automatically, politely escalate it.
    - When greeting or closing, keep it brief and professional.
    """
    try:
        past_memories = retrieve_memeory(phone_number, user_text)
        if past_memories and isinstance(past_memories, list) and len(past_memories) >0:
            memory_context = "/n--------------------------------/n".join(past_memories) if past_memories else "None"
        else:
            memory_context = None
    except Exception as e:
        print(f"Failed{e}")
    
    prompt = f"{system_prompt}. The customer said: {user_text}\n"
    if sap_info:
        prompt += f"Customer info from SAP: {sap_info}\n"
    if memory_context:
        prompt += f"Below is a summary of your previous conversations with this same customer. Use this as your memory to provide continuity and avoid repeating questions PAST INTERACTIONS: {memory_context}"
    prompt += "Provide a helpful response following the context and rules that where given"
    
    response = client.chat.completions.create(
         model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
        )
    
    prompt = prompt[-8000:]
    
    reply = response.choices[0].message.content
    store_memory(phone_number, user_text, reply)
    print(memory_collection.count())
    print(memory_collection.peek())
    return reply

def live_conversation(phone_number: int):
   log_path = create_new_log_file()
   conversation_log = load_conversation_log(log_path)
   sap_data = fetch_sap_info(phone_number)
   
   while True:
        try:
            print("ctrl C to exit")
            audio_file = speech_toggle()
            text = transcribe_audio(audio_file)

            if text is None:
                print("No transcription available, try again.")
                continue

            if text.strip().lower() == "quit":
                print("Ending Convo")
                conversation_log["resolved"] = True
                save_conversation_log(log_path, conversation_log)
                break
            
            reply = ai_response(text, sap_data, phone_number)
            #if text_to_speech(reply):
            #    text_to_speech(reply)
            text_to_speech_fallback(reply)

            conversation_log["entries"].append({
                "input": text,
                "reply": reply,
                "phone": phone_number
            })
            save_conversation_log(log_path, conversation_log)
        
        except KeyboardInterrupt:
            print("exit")
            conversation_log["resolved"] = True
            save_conversation_log(log_path, conversation_log)
            break


    

print(conversation_log)
