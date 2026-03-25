import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

models_to_try = [
    'gemini-flash-latest',
    'gemini-pro-latest',
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-2.5-flash',
    'gemini-1.5-flash', # Let's include it just in case
]

for model_name in models_to_try:
    print(f"\nTrying {model_name}...")
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hi", timeout=10)
        print(f"  Success! Response: {response.text[:50]}")
        break
    except Exception as e:
        print(f"  Failed for {model_name}: {e}")
