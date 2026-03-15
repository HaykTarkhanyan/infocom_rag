"""Parse Telegram Desktop 'Export Chat History' JSON and ingest into Weaviate."""

import json
import sys
from pathlib import Path

import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances

from config import (
    WEAVIATE_URL,
    COLLECTION_NAME,
    CHUNKING_STRATEGY,
    CHUNK_WINDOW_SIZE,
    CHUNK_OVERLAP,
    THREAD_MAX_MESSAGES,
    MIN_MESSAGE_LENGTH,
    INGEST_BATCH_SIZE,
    DISTANCE_METRIC,
    HNSW_EF_CONSTRUCTION,
    HNSW_MAX_CONNECTIONS,
)
from embeddings import EmbeddingModel


DISTANCE_MAP = {
    "cosine": VectorDistances.COSINE,
    "dot": VectorDistances.DOT,
    "l2": VectorDistances.L2_SQUARED,
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def extract_text(raw_text) -> str:
    """Extract plain text from Telegram's polymorphic text field."""
    if isinstance(raw_text, list):
        parts = []
        for part in raw_text:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "".join(parts)
    return raw_text if isinstance(raw_text, str) else ""


def parse_telegram_export(path: str) -> list[dict]:
    """Parse result.json from Telegram Desktop export."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = []
    chat_name = data.get("name", "Unknown Chat")

    for msg in data.get("messages", []):
        if msg.get("type") != "message":
            continue

        text = extract_text(msg.get("text", "")).strip()
        if not text or len(text) < MIN_MESSAGE_LENGTH:
            continue

        messages.append({
            "text": text,
            "sender": msg.get("from", msg.get("actor", "Unknown")),
            "date": msg.get("date", ""),
            "message_id": msg.get("id", 0),
            "reply_to_message_id": msg.get("reply_to_message_id"),
            "chat_name": chat_name,
        })

    return messages


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------


def chunk_single_message(messages: list[dict]) -> list[dict]:
    """Each message is its own chunk (default)."""
    chunks = []
    for m in messages:
        chunks.append({
            "text": f"[{m['date']}] {m['sender']}: {m['text']}",
            "sender": m["sender"],
            "date": m["date"],
            "message_id": m["message_id"],
            "reply_to_message_id": m.get("reply_to_message_id"),
            "chat_name": m["chat_name"],
            "chunk_type": "single_message",
        })
    return chunks


def chunk_sliding_window(
    messages: list[dict],
    window_size: int = CHUNK_WINDOW_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Group consecutive messages into overlapping windows."""
    chunks = []
    step = max(1, window_size - overlap)

    for i in range(0, len(messages), step):
        window = messages[i : i + window_size]
        if not window:
            break

        combined_text = "\n".join(
            f"[{m['date']}] {m['sender']}: {m['text']}" for m in window
        )
        chunks.append({
            "text": combined_text,
            "sender": ", ".join(sorted({m["sender"] for m in window})),
            "date": window[0]["date"],
            "message_id": window[0]["message_id"],
            "reply_to_message_id": None,
            "chat_name": window[0]["chat_name"],
            "chunk_type": f"sliding_window_{window_size}_{overlap}",
        })

    return chunks


def chunk_conversation_thread(
    messages: list[dict],
    max_per_thread: int = THREAD_MAX_MESSAGES,
) -> list[dict]:
    """Group messages by reply chains into conversation threads."""
    id_to_msg = {m["message_id"]: m for m in messages}
    visited = set()
    threads: list[list[dict]] = []

    # Build adjacency: parent -> children
    children: dict[int, list[int]] = {}
    for m in messages:
        rid = m.get("reply_to_message_id")
        if rid and rid in id_to_msg:
            children.setdefault(rid, []).append(m["message_id"])

    def collect_thread(root_id: int) -> list[dict]:
        """BFS to collect a thread starting from root."""
        from collections import deque
        queue = deque([root_id])
        thread = []
        while queue:
            mid = queue.popleft()
            if mid in visited or mid not in id_to_msg:
                continue
            visited.add(mid)
            thread.append(id_to_msg[mid])
            for child_id in children.get(mid, []):
                queue.append(child_id)
        return sorted(thread, key=lambda m: m["message_id"])

    # Find root messages (not a reply, or reply target not in export)
    for m in messages:
        if m["message_id"] in visited:
            continue
        rid = m.get("reply_to_message_id")
        if rid and rid in id_to_msg:
            continue  # not a root
        thread = collect_thread(m["message_id"])
        if thread:
            threads.append(thread)

    # Pick up any orphans
    for m in messages:
        if m["message_id"] not in visited:
            visited.add(m["message_id"])
            threads.append([m])

    # Convert threads to chunks, splitting long threads
    chunks = []
    for thread in threads:
        for i in range(0, len(thread), max_per_thread):
            sub = thread[i : i + max_per_thread]
            combined_text = "\n".join(
                f"[{m['date']}] {m['sender']}: {m['text']}" for m in sub
            )
            chunks.append({
                "text": combined_text,
                "sender": ", ".join(sorted({m["sender"] for m in sub})),
                "date": sub[0]["date"],
                "message_id": sub[0]["message_id"],
                "reply_to_message_id": None,
                "chat_name": sub[0]["chat_name"],
                "chunk_type": "conversation_thread",
            })

    return chunks


CHUNKING_FUNCTIONS = {
    "single_message": chunk_single_message,
    "sliding_window": chunk_sliding_window,
    "conversation_thread": chunk_conversation_thread,
}


def apply_chunking(messages: list[dict], strategy: str = CHUNKING_STRATEGY) -> list[dict]:
    """Apply the configured chunking strategy."""
    func = CHUNKING_FUNCTIONS.get(strategy)
    if func is None:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. "
            f"Choose from: {list(CHUNKING_FUNCTIONS.keys())}"
        )
    return func(messages)


# ---------------------------------------------------------------------------
# Weaviate operations
# ---------------------------------------------------------------------------


def create_collection(client: weaviate.WeaviateClient):
    """Create the Weaviate collection (drops existing one)."""
    if client.collections.exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting and recreating.")
        client.collections.delete(COLLECTION_NAME)

    distance = DISTANCE_MAP.get(DISTANCE_METRIC, VectorDistances.COSINE)

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=distance,
            ef_construction=HNSW_EF_CONSTRUCTION,
            max_connections=HNSW_MAX_CONNECTIONS,
        ),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="sender", data_type=DataType.TEXT),
            Property(name="date", data_type=DataType.TEXT),
            Property(name="message_id", data_type=DataType.INT),
            Property(name="reply_to_message_id", data_type=DataType.INT),
            Property(name="chat_name", data_type=DataType.TEXT),
            Property(name="chunk_type", data_type=DataType.TEXT),
        ],
    )
    print(f"Created collection '{COLLECTION_NAME}' (distance={DISTANCE_METRIC}).")


def ingest_chunks(
    client: weaviate.WeaviateClient,
    chunks: list[dict],
    embedder: EmbeddingModel,
):
    """Batch-insert chunks with embeddings into Weaviate."""
    collection = client.collections.get(COLLECTION_NAME)

    total = len(chunks)
    for i in range(0, total, INGEST_BATCH_SIZE):
        batch = chunks[i : i + INGEST_BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = embedder.embed_documents(texts)

        with collection.batch.dynamic() as weaviate_batch:
            for chunk, vec in zip(batch, vectors):
                props = {
                    "text": chunk["text"],
                    "sender": chunk["sender"],
                    "date": chunk["date"],
                    "message_id": chunk["message_id"],
                    "chat_name": chunk["chat_name"],
                    "chunk_type": chunk.get("chunk_type", "single_message"),
                }
                if chunk.get("reply_to_message_id") is not None:
                    props["reply_to_message_id"] = chunk["reply_to_message_id"]

                weaviate_batch.add_object(properties=props, vector=vec)

        done = min(i + INGEST_BATCH_SIZE, total)
        print(f"Ingested {done}/{total} chunks")

    print("Ingestion complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path-to-result.json> [chunking_strategy]")
        print(f"  Strategies: {list(CHUNKING_FUNCTIONS.keys())}")
        sys.exit(1)

    json_path = sys.argv[1]
    strategy = sys.argv[2] if len(sys.argv) > 2 else CHUNKING_STRATEGY

    if not Path(json_path).exists():
        print(f"File not found: {json_path}")
        sys.exit(1)

    print(f"Parsing Telegram export: {json_path}")
    messages = parse_telegram_export(json_path)
    print(f"Found {len(messages)} text messages.")

    if not messages:
        print("No messages to ingest.")
        sys.exit(0)

    print(f"Applying chunking strategy: {strategy}")
    chunks = apply_chunking(messages, strategy)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model (this may take a moment on first run)...")
    embedder = EmbeddingModel()

    print(f"Connecting to Weaviate at {WEAVIATE_URL}")
    from urllib.parse import urlparse
    parsed = urlparse(WEAVIATE_URL)
    client = weaviate.connect_to_local(
        host=parsed.hostname or "localhost",
        port=parsed.port or 8080,
    )

    try:
        create_collection(client)
        ingest_chunks(client, chunks, embedder)
    finally:
        client.close()


if __name__ == "__main__":
    main()
