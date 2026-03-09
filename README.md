# Telegram RAG Chatbot

A RAG chatbot that ingests Telegram chat history and answers questions using:
- **Metric-AI/armenian-text-embeddings-1** for embeddings
- **Weaviate** as the vector database
- **Gemini 2.5 Flash** for answer generation
- **python-telegram-bot** for the Telegram interface

## Setup

### 1. Prerequisites
- Python 3.11+
- Docker & Docker Compose

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:
- `TELEGRAM_BOT_TOKEN` — get from [@BotFather](https://t.me/BotFather)
- `GEMINI_API_KEY` — get from [Google AI Studio](https://aistudio.google.com/)

### 4. Start Weaviate

```bash
docker compose up -d
```

### 5. Export Telegram chat

In Telegram Desktop: **Settings → Advanced → Export chat history → JSON format**

This produces a `result.json` file.

### 6. Ingest data

```bash
python ingest.py path/to/result.json
```

This parses the Telegram export, embeds all messages, and stores them in Weaviate.

### 7. Run the bot

```bash
python bot.py
```

Open your bot in Telegram and start asking questions!

## Architecture

```
result.json → ingest.py → [Armenian Embeddings] → Weaviate
                                                      ↓
User Question → bot.py → [Armenian Embeddings] → Vector Search
                                                      ↓
                                              Retrieved Context
                                                      ↓
                                                Gemini 2.5 Flash
                                                      ↓
                                                   Answer
```

## Configuration

Edit `config.py` to adjust:
- `TOP_K` — number of messages to retrieve (default: 10)
- `COLLECTION_NAME` — Weaviate collection name
- Embedding model settings
