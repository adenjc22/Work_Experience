import pandas as pd
import json
import whisper 
from openai import OpenAI

client = OpenAI(api_key="api_key")

sap_data = pd.read_excel("demo_sap.xlsx")

model = whisper.load_model("small")

conversation_log = []

def transcrie_audio(file_path: str) -> str:
    print("Transcribing {file_path}")
    result = model.transcribe(file_path)
    text = result['text']
    print(f"Transcribed text: {text}")
    return text

transcrie_audio("complaint.wav")

