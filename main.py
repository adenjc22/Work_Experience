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

def fetch_sap_info(phone_number: str) -> dict:
    result = sap_data[sap_data['CustomerPhone'] == phone_number]
    if not result.empty:
        return result.to_dict(orient='records')[0]
    return None

def ai_response(text: str, sap_info: dict = None) -> str:
    prompt = f"You are a customer service assistant.\nCustomer said: {text}\n"
    if sap_info:
        prompt += f"Customer info from SAP: {sap_info}\n"
    prompt += "Provide a helpful response."
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

