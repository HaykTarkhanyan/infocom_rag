# Telegram RAG Chatbot

A RAG chatbot that ingests Telegram chat history and answers questions using:
- **Metric-AI/armenian-text-embeddings-1** for embeddings (768-dim, Armenian-optimized)
- **Weaviate** as the vector database (Docker)
- **Gemini 2.5 Flash** for answer generation
- **Two interfaces**: Telegram bot + Streamlit web UI

Supports Armenian, English, and Russian in mixed-language settings.

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env   # fill in TELEGRAM_BOT_TOKEN and GEMINI_API_KEY

# 3. Start Weaviate
docker compose up -d

# 4. Export your Telegram chat (Telegram Desktop → Settings → Advanced → Export chat history → JSON)

# 5. Ingest
python ingest.py path/to/result.json                    # default: single_message chunking
python ingest.py path/to/result.json sliding_window     # or: sliding_window / conversation_thread

# 6. Run
python bot.py              # Telegram bot
streamlit run ui.py        # Web UI with live hyperparameter tuning
```

## Chunking Strategies

| Strategy | Description | Best for |
|---|---|---|
| `single_message` | Each message = one chunk | Large chats, precise retrieval |
| `sliding_window` | N consecutive messages with overlap | Group chats with flowing conversations |
| `conversation_thread` | Reply chains grouped together | Chats where people use the reply feature |

## Configuration

All settings are configurable via environment variables in `.env`. See `.env.example` for the full list, or `REPORT.md` for detailed documentation.

Key settings: `TOP_K`, `MAX_DISTANCE`, `CHUNKING_STRATEGY`, `GEMINI_MODEL`, `GEMINI_TEMPERATURE`.

The Streamlit UI also lets you tune retrieval and generation parameters in real-time via the sidebar.

## Documentation

See **[REPORT.md](REPORT.md)** for the full technical report covering architecture, data flow, chunking strategies, hyperparameter tuning guide, and performance considerations.
