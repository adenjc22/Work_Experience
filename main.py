import pandas as pd
import json
import whisper 
from openai import OpenAI
import warnings
import soundfile as sf
import sounddevice as sd
import numpy as np
import wave
import websockets
import pyttsx3
warnings.filterwarnings("ignore")



client = OpenAI(api_key="YOUR_API_KEY_HERE")


client = OpenAI(api_key="API_KEY")
sap_data = pd.read_excel("demo_sap.xlsx")
model = whisper.load_model("small")

conversation_log = []

def transcribe_audio(file_path: str) -> str:
    print(f"Transcribing {file_path}")
    result = model.transcribe(file_path)
    text = result['text']
    print(f"Transcribed text: {text}")
    return text

def text_to_speech(text, audio_file, voice="alloy"):
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text
    )

    with open(audio_file, "wb") as f:
        f.write(response.read())

    data, samplerate = sf.read(audio_file)
    sd.play(data, samplerate)
    sd.wait()

def text_to_speech_fallback(text, audio_file):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

    with open(audio_file, "wb") as f:
        engine.save_to_file(text, audio_file)



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


    

def middleware_call(phone_number: str):

    audio = speech_to_text("input.wav", duration=5)
    text = transcribe_audio(audio)
    sap_info = fetch_sap_info(phone_number)

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

'''

middleware_call(447911000012,"delivery_inquiry.wav")
middleware_call(447911000018,"complaint.wav")
middleware_call(447911000032,"human_escalation.wav")
middleware_call(447911000082,"order_status_check.wav")
middleware_call(447911000085,"info_request.wav")
'''

print(conversation_log)
