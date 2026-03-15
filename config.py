import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys & Services ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

# --- Embedding Model ---
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "Metric-AI/armenian-text-embeddings-1"
)
EMBEDDING_DIM = 768
MAX_TOKENS = 512

# --- Weaviate ---
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "TelegramMessages")

# --- Chunking Strategy ---
# Options: "single_message", "sliding_window", "conversation_thread"
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "single_message")

# sliding_window: group N consecutive messages into one chunk
CHUNK_WINDOW_SIZE = int(os.getenv("CHUNK_WINDOW_SIZE", "5"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "2"))

# conversation_thread: group messages by reply chains
# max messages per thread chunk before splitting
THREAD_MAX_MESSAGES = int(os.getenv("THREAD_MAX_MESSAGES", "20"))

# Minimum message length to keep (filters out very short messages like "ok", "yes")
MIN_MESSAGE_LENGTH = int(os.getenv("MIN_MESSAGE_LENGTH", "0"))

# --- Ingestion ---
INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "64"))

# --- RAG Retrieval ---
TOP_K = int(os.getenv("TOP_K", "10"))

# Distance metric: "cosine", "dot", "l2"
DISTANCE_METRIC = os.getenv("DISTANCE_METRIC", "cosine")

# Maximum distance threshold — discard results above this (0.0 = identical, 2.0 = opposite)
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "1.0"))

# --- HNSW Index ---
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "128"))
HNSW_MAX_CONNECTIONS = int(os.getenv("HNSW_MAX_CONNECTIONS", "64"))
HNSW_EF = int(os.getenv("HNSW_EF", "256"))

# --- Gemini Generation ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024"))
GEMINI_TOP_P = float(os.getenv("GEMINI_TOP_P", "0.95"))
GEMINI_TOP_K_SAMPLING = int(os.getenv("GEMINI_TOP_K_SAMPLING", "40"))

# --- System Prompt ---
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant that answers questions based on Telegram chat history. "
    "You will be given relevant messages from the chat as context. "
    "Answer the user's question based on the provided context. "
    "If the context doesn't contain enough information, say so honestly. "
    "You can respond in Armenian, English, or Russian — match the language of the user's question. "
    "Keep answers concise and relevant.",
)
