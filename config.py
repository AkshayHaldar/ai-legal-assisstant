import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY   = os.getenv("TAVILY_API_KEY", "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") # User provided key
MODEL_NAME       = os.getenv("MODEL_NAME", "gemini-flash-latest")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/processed/legal_index")
AUDIT_LOG_PATH   = os.getenv("AUDIT_LOG_PATH",   "logs/audit.jsonl")

# HuggingFace embedding model — free, runs locally, no API key needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
TOP_K_RETRIEVAL = 4

# Initialise global Gemini client (google-genai v1 SDK)
client = genai.Client(api_key=GOOGLE_API_KEY)
