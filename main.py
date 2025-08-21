import pandas as pd
import json
import whisper 
from openai import OpenAI
import warnings
warnings.filterwarnings("ignore")



client = OpenAI(api_key="api key")

sap_data = pd.read_excel("demo_sap.xlsx")

model = whisper.load_model("small")

conversation_log = []

def transcribe_audio(file_path: str) -> str:
    print(f"Transcribing {file_path}")
    result = model.transcribe(file_path)
    text = result['text']
    print(f"Transcribed text: {text}")
    return text

def fetch_sap_info(phone_number: int) -> dict:
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
         model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
        )
    return response.choices[0].message.content
    

def middleware_call(phone_number: str, audio_file: str):

    text = transcribe_audio(audio_file)
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
    
    # Step 5: Log conversation
    log_entry = {
        "phone": phone_number,
        "audio_file": audio_file,
        "input_text": text,
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

middleware_call(447911000012,"delivery_inquiry.wav")
middleware_call(447911000018,"complaint.wav")
middleware_call(447911000032,"human_escalation.wav")
middleware_call(447911000082,"order_status_check.wav")
middleware_call(447911000085,"info_request.wav")

print(conversation_log)