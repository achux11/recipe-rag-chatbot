import os

# === API KEYS ===
# Set these via environment variables for safety.
# On Windows PowerShell, e.g.:
#   $env:OPENAI_API_KEY = "your-key-here"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOCUMENTS_DIR = os.path.join(BASE_DIR, "data")       # put your recipe PDFs/text here
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")  # Chroma persistence

COLLECTION_NAME = "recipe_rag_collection"

# === MODELS ===
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL_NAME = "gpt-4.1-mini"  # can change if needed

