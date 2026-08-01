# Telegram RAG Chatbot — Technical Report

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
   - [Embedding Model](#embedding-model)
   - [Vector Database (Weaviate)](#vector-database-weaviate)
   - [Ingestion Pipeline](#ingestion-pipeline)
   - [RAG Engine](#rag-engine)
   - [Telegram Bot](#telegram-bot)
   - [Streamlit UI](#streamlit-ui)
4. [Chunking Strategies](#chunking-strategies)
5. [Hyperparameters Reference](#hyperparameters-reference)
6. [Data Flow](#data-flow)
7. [File Structure](#file-structure)
8. [Configuration Guide](#configuration-guide)
9. [Performance Considerations](#performance-considerations)
10. [Limitations & Future Work](#limitations--future-work)

---

## Overview

This project is a **Retrieval-Augmented Generation (RAG)** chatbot that ingests Telegram chat history exported as JSON and allows users to ask questions about the conversations. It combines:

- **Metric-AI/armenian-text-embeddings-1** — a state-of-the-art Armenian text embedding model (768-dim, based on multilingual-e5-base)
- **Weaviate** — an open-source vector database running in Docker
- **Google Gemini** — for natural language answer generation
- **Two interfaces**: a Telegram bot and a Streamlit web UI

The system supports Armenian, English, and Russian in a mixed-language setting.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        INGESTION PHASE                           │
│                                                                  │
│  result.json ─→ parse_telegram_export() ─→ Raw Messages          │
│                                               │                  │
│                                    apply_chunking()              │
│                                       │                          │
│                          ┌────────────┼────────────┐             │
│                          │            │            │             │
│                   single_message  sliding_window  thread         │
│                          │            │            │             │
│                          └────────────┼────────────┘             │
│                                       │                          │
│                          Armenian Text Embeddings                │
│                          (Metric-AI, 768-dim)                    │
│                                       │                          │
│                                   Weaviate                       │
│                              (HNSW Index)                        │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                         QUERY PHASE                              │
│                                                                  │
│  User Question ─→ embed_query() ─→ Vector Search (Weaviate)     │
│                                          │                       │
│                                   Top-K Chunks                   │
│                                   (filtered by max_distance)     │
│                                          │                       │
│                                   Format Context                 │
│                                          │                       │
│                               Gemini 2.5 Flash                   │
│                          (system prompt + context + query)        │
│                                          │                       │
│                                      Answer                      │
│                                    ┌─────┴─────┐                 │
│                              Telegram Bot   Streamlit UI         │
└──────────────────────────────────────────────────────────────────┘
```

---

## Components

### Embedding Model

**File:** `embeddings.py`

Uses `Metric-AI/armenian-text-embeddings-1`, a fine-tuned version of Microsoft's `intfloat/multilingual-e5-base`.

| Property | Value |
|---|---|
| Dimensions | 768 |
| Max tokens | 512 |
| Base model | intfloat/multilingual-e5-base (XLM-RoBERTa) |
| Training data | 2M+ Reddit pairs translated to Armenian via Gemma 2 27b-it |
| Retrieval accuracy | 89% (outperforms OpenAI embeddings on Armenian benchmarks) |
| License | Apache 2.0 |

**Key implementation details:**
- Uses `"query: "` prefix for search queries and `"passage: "` prefix for documents (E5 convention)
- Applies average pooling over the last hidden states, masked by attention
- Normalizes output embeddings to unit vectors (L2 norm)
- Runs inference with `torch.no_grad()` for efficiency

```python
class EmbeddingModel:
    def embed_query(text) -> list[float]       # prefix="query"
    def embed_documents(texts) -> list[list[float]]  # prefix="passage"
```

### Vector Database (Weaviate)

**File:** `docker-compose.yml`

Weaviate runs as a Docker container with:
- **HNSW index** for approximate nearest neighbor search
- **No built-in vectorizer** — we provide our own embeddings
- Persistent volume for data storage
- Anonymous access enabled (for local development)

The collection schema stores:
- `text` (TEXT) — the chunk content
- `sender` (TEXT) — message author(s)
- `date` (TEXT) — timestamp
- `message_id` (INT) — Telegram message ID
- `reply_to_message_id` (INT) — for thread tracking
- `chat_name` (TEXT) — source chat name
- `chunk_type` (TEXT) — which chunking strategy produced this chunk

### Ingestion Pipeline

**File:** `ingest.py`

The ingestion pipeline has three stages:

#### Stage 1: Parsing

Reads Telegram Desktop's `result.json` export format. Handles:
- Polymorphic `text` field (plain string or array of mixed text/entity objects)
- `from` vs `actor` sender fields (regular messages vs service messages)
- Filters out non-message entries (`type != "message"`)
- Configurable minimum message length filter

#### Stage 2: Chunking

Three strategies are available (see [Chunking Strategies](#chunking-strategies) below).

#### Stage 3: Embedding & Storage

- Processes chunks in configurable batch sizes
- Embeds all chunk texts using the Armenian embeddings model
- Batch-inserts into Weaviate with pre-computed vectors

### RAG Engine

**File:** `rag.py`

The `RAG` class encapsulates the full retrieve-and-generate pipeline:

1. **Retrieve**: Embeds the query, performs `near_vector` search on Weaviate, filters by `max_distance`
2. **Format**: Concatenates retrieved chunks into a context string
3. **Generate**: Sends `system_prompt + context + query` to Gemini

All parameters (top_k, max_distance, temperature, model, system_prompt) are settable at runtime, enabling live tuning from the Streamlit UI.

### Telegram Bot

**File:** `bot.py`

A simple `python-telegram-bot` application:
- `/start` command with welcome message
- All text messages routed through the RAG pipeline
- Shows typing indicator while processing
- Error handling with user-friendly messages

### Streamlit UI

**File:** `ui.py`

A web-based chat interface with:
- **Sidebar controls** for all RAG hyperparameters (applied in real-time)
- **Chat history** persisted in session state
- **Expandable source panels** showing retrieved chunks with distance scores
- **Cached RAG instance** (embedding model loaded once)

Run with: `streamlit run ui.py`

---

## Chunking Strategies

### 1. `single_message` (default)

Each Telegram message becomes its own chunk.

```
Chunk 1: "[2024-01-15] Alice: Ինչպես ես?"
Chunk 2: "[2024-01-15] Bob: Լdelays delays, շdelays delays:"
Chunk 3: "[2024-01-15] Alice: Հdelays delays!"
```

**Pros:** Precise retrieval, no noise from unrelated messages
**Cons:** Loses conversational context, short messages may not embed well
**Best for:** Large chats where conversations are not heavily threaded

### 2. `sliding_window`

Groups N consecutive messages into overlapping windows.

```
Window size=3, overlap=1:
Chunk 1: messages [1, 2, 3]
Chunk 2: messages [3, 4, 5]
Chunk 3: messages [5, 6, 7]
```

**Pros:** Preserves local conversational context, overlapping ensures no context is lost at boundaries
**Cons:** Larger chunks = more noise per retrieval, increased storage
**Best for:** Group chats with flowing conversations

**Parameters:**
- `CHUNK_WINDOW_SIZE` (default: 5) — messages per window
- `CHUNK_OVERLAP` (default: 2) — overlap between consecutive windows

### 3. `conversation_thread`

Groups messages by reply chains (using `reply_to_message_id`).

```
Thread 1: Alice asks → Bob replies → Alice follows up
Thread 2: Carol starts new topic → Dave responds
Orphan:   Eve's standalone message
```

**Pros:** Semantically coherent chunks, captures full Q&A exchanges
**Cons:** Depends on users actually using the reply feature; orphan messages become single-message chunks
**Best for:** Chats where people frequently use the reply feature

**Parameters:**
- `THREAD_MAX_MESSAGES` (default: 20) — split threads longer than this

---

## Hyperparameters Reference

### Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — | Telegram bot token from @BotFather |
| `GEMINI_API_KEY` | — | Google AI Studio API key |
| `WEAVIATE_URL` | `http://localhost:8080` | Weaviate server URL |

### Embedding & Model (`config.py` / `.env`)

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `Metric-AI/armenian-text-embeddings-1` | HuggingFace model ID |
| `EMBEDDING_DIM` | `768` | Embedding vector dimensionality |
| `MAX_TOKENS` | `512` | Max input sequence length for the embedding model |

### Chunking

| Variable | Default | Description |
|---|---|---|
| `CHUNKING_STRATEGY` | `single_message` | One of: `single_message`, `sliding_window`, `conversation_thread` |
| `CHUNK_WINDOW_SIZE` | `5` | Number of messages per sliding window |
| `CHUNK_OVERLAP` | `2` | Overlap between consecutive sliding windows |
| `THREAD_MAX_MESSAGES` | `20` | Max messages per conversation thread chunk |
| `MIN_MESSAGE_LENGTH` | `0` | Discard messages shorter than this (characters) |

### Ingestion

| Variable | Default | Description |
|---|---|---|
| `INGEST_BATCH_SIZE` | `64` | Number of chunks embedded and inserted per batch |

### HNSW Index (Weaviate)

| Variable | Default | Description |
|---|---|---|
| `HNSW_EF_CONSTRUCTION` | `128` | Build-time search depth. Higher = better recall, slower indexing |
| `HNSW_MAX_CONNECTIONS` | `64` | Max edges per node. Higher = better recall, more memory |
| `HNSW_EF` | `256` | Query-time search depth. Higher = better recall, slower queries |
| `DISTANCE_METRIC` | `cosine` | One of: `cosine`, `dot`, `l2` |

### RAG Retrieval

| Variable | Default | Description |
|---|---|---|
| `TOP_K` | `10` | Number of chunks to retrieve per query |
| `MAX_DISTANCE` | `1.0` | Discard chunks with distance above this threshold |

### Gemini Generation

| Variable | Default | Description |
|---|---|---|
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model to use |
| `GEMINI_TEMPERATURE` | `0.3` | Sampling temperature (0 = deterministic, 2 = very creative) |
| `GEMINI_MAX_OUTPUT_TOKENS` | `1024` | Maximum response length |
| `GEMINI_TOP_P` | `0.95` | Nucleus sampling threshold |
| `GEMINI_TOP_K_SAMPLING` | `40` | Top-K sampling parameter |
| `SYSTEM_PROMPT` | *(see config.py)* | System instruction for Gemini |

---

## Data Flow

### Ingestion Flow

```
1. User runs: python ingest.py result.json [strategy]
2. parse_telegram_export(result.json)
   └─ Reads JSON, filters type="message", extracts text (handles polymorphic field)
   └─ Applies MIN_MESSAGE_LENGTH filter
   └─ Returns list of message dicts
3. apply_chunking(messages, strategy)
   └─ single_message: 1 message → 1 chunk
   └─ sliding_window: N messages → 1 chunk (with overlap)
   └─ conversation_thread: reply chains → 1 chunk each
4. For each batch of INGEST_BATCH_SIZE chunks:
   └─ EmbeddingModel.embed_documents(texts) → 768-dim vectors
   └─ Weaviate batch insert (properties + vector)
```

### Query Flow

```
1. User sends question via Telegram / Streamlit
2. EmbeddingModel.embed_query(question) → 768-dim vector
3. Weaviate near_vector search (top_k results)
4. Filter by max_distance
5. Format retrieved chunks into context string
6. Send to Gemini: system_prompt + context + question
7. Return generated answer (+ sources in Streamlit UI)
```

---

## File Structure

```
telegram-rag-chatbot/
├── config.py           # All configuration and hyperparameters
├── embeddings.py       # Armenian text embedding model wrapper
├── ingest.py           # Telegram JSON parser + chunking + Weaviate ingestion
├── rag.py              # RAG retrieval + Gemini generation engine
├── bot.py              # Telegram bot interface
├── ui.py               # Streamlit web UI with live hyperparameter tuning
├── docker-compose.yml  # Weaviate Docker service
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Python/IDE ignores
├── README.md           # Quick-start guide
└── REPORT.md           # This report
```

---

## Configuration Guide

### Tuning for Quality

For best retrieval quality:
```env
CHUNKING_STRATEGY=conversation_thread
TOP_K=15
MAX_DISTANCE=0.6
HNSW_EF_CONSTRUCTION=256
HNSW_MAX_CONNECTIONS=128
HNSW_EF=512
GEMINI_TEMPERATURE=0.2
```

### Tuning for Speed

For faster responses with acceptable quality:
```env
CHUNKING_STRATEGY=single_message
TOP_K=5
HNSW_EF_CONSTRUCTION=64
HNSW_MAX_CONNECTIONS=32
HNSW_EF=128
GEMINI_MODEL=gemini-2.0-flash-lite
```

### Tuning for Large Chats (100k+ messages)

```env
CHUNKING_STRATEGY=sliding_window
CHUNK_WINDOW_SIZE=3
CHUNK_OVERLAP=1
MIN_MESSAGE_LENGTH=10
INGEST_BATCH_SIZE=128
TOP_K=10
MAX_DISTANCE=0.7
```

---

## Performance Considerations

### Embedding Model
- The model has 278M parameters and runs on CPU by default
- First load downloads ~1GB from HuggingFace (cached after that)
- Embedding speed: ~50-100 messages/sec on CPU, much faster on GPU
- For GPU acceleration, ensure CUDA-compatible PyTorch is installed

### Weaviate
- HNSW index provides sub-millisecond search at scale
- Memory usage scales with number of vectors × dimensions (768 floats × 4 bytes each)
- For 100k messages: ~300MB of vector data
- The `ef_construction` and `max_connections` params trade indexing speed for recall quality

### Gemini API
- `gemini-2.5-flash` offers the best speed/quality tradeoff
- `gemini-2.5-pro` is available for higher quality at higher latency and cost
- Rate limits apply per the Google AI Studio plan

### Chunking Impact
- `single_message`: fastest ingestion, smallest storage footprint
- `sliding_window`: ~(window_size / step) × more chunks than single_message
- `conversation_thread`: depends on reply density; typically 30-60% fewer chunks than single_message

---

## Limitations & Future Work

### Current Limitations
- **No incremental ingestion** — re-ingesting drops and recreates the entire collection
- **No media support** — only text messages are indexed (photos, videos, stickers are skipped)
- **No conversation memory** — each query is independent (no multi-turn chat with the bot)
- **CPU-only embedding** — no GPU auto-detection; users must configure CUDA manually
- **Single chat export** — does not handle full-account exports with multiple chats

### Potential Improvements
- **Incremental ingestion** — check existing message IDs before inserting, add new messages only
- **Hybrid search** — combine vector search with BM25 keyword search in Weaviate
- **Reranking** — add a cross-encoder reranker after initial retrieval for better precision
- **Multi-turn conversation** — maintain chat history in the bot for follow-up questions
- **Media captions** — extract and index captions from photos/videos
- **Date-aware retrieval** — allow queries like "what did Alice say last week?"
- **User filtering** — restrict retrieval to messages from specific senders
- **Evaluation framework** — automated retrieval quality metrics (MRR, NDCG) on labeled query sets
