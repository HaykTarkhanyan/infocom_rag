import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

EMBEDDING_MODEL_NAME = "Metric-AI/armenian-text-embeddings-1"
EMBEDDING_DIM = 768
MAX_TOKENS = 512

COLLECTION_NAME = "TelegramMessages"

# RAG settings
TOP_K = 10
