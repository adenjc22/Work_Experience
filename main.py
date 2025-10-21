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


def text_to_speech(text, voice="alloy"):
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text
    ) as response:
        response.stream_to_playback()



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

def speech_to_text(filename, duration=5, sample_rate=SAMPLE_RATE):
    print("recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2)
    sd.wait()
    print("done recording")

    #save as wav file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((recording * 32767).astype(np.int16).tobytes())

    return filename

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

def ai_response(text: str, sap_info: dict = None) -> str:
    prompt = f"You are a customer service assistant. Do not give information that is not specfically asked for. \nCustomer said: {text}\n"
    if sap_info:
        prompt += f"Customer info from SAP: {sap_info}\n"
    prompt += "Provide a helpful response."
    
    response = client.chat.completions.create(
         model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
        )
    return response.choices[0].message.content

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
            
            reply = ai_response(text, sap_data)
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

    

def middleware_call(phone_number: str):

    sap_info = fetch_sap_info(phone_number)

    audio = speech_toggle("input.wav")
    text = transcribe_audio(audio)


    reply = ai_response(text, sap_info)
    
    #Determine if AI can resolve
    if sap_info is None: 
        resolved = False
        escalation_message = "This query requires human support. Escalating to agent..."
        final_reply = escalation_message
    else:
        resolved = True
        final_reply = reply  # AI can resolve using SAP info

    text_to_speech_fallback(final_reply, "output.wav")
    
    # Step 5: Log conversation
    log_entry = {
        "phone": phone_number,
        "input_audio": audio,
        "sap_info": sap_info,
        "ai_response": reply,
        "resolved": resolved
    }
    
    conversation_log.append(log_entry)
    
    # Step 6: Save log to JSON
    with open("conversation_log.json", "w") as f:
        json.dump(conversation_log, f, indent=4)
    
    # Step 7: Return final reply (AI or escalation message)
    print(final_reply)
    return final_reply


print(conversation_log)
