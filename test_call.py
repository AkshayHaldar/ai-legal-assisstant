import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("Starting test call with 'gemini-flash-latest'...")
model = genai.GenerativeModel('gemini-flash-latest')
try:
    response = model.generate_content("Hi", timeout=10)
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
