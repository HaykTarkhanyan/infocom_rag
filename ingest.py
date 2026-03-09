"""Parse Telegram Desktop 'Export Chat History' JSON and ingest into Weaviate."""

import json
import sys
from datetime import datetime
from pathlib import Path

import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.data import DataObject

from config import WEAVIATE_URL, COLLECTION_NAME, EMBEDDING_DIM
from embeddings import EmbeddingModel

BATCH_SIZE = 64


def parse_telegram_export(path: str) -> list[dict]:
    """Parse result.json from Telegram Desktop export."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = []
    chat_name = data.get("name", "Unknown Chat")

    for msg in data.get("messages", []):
        if msg.get("type") != "message":
            continue

        # Extract text - can be a string or list of mixed text/entities
        raw_text = msg.get("text", "")
        if isinstance(raw_text, list):
            parts = []
            for part in raw_text:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(part.get("text", ""))
            text = "".join(parts)
        else:
            text = raw_text

        text = text.strip()
        if not text:
            continue

        sender = msg.get("from", msg.get("actor", "Unknown"))
        date_str = msg.get("date", "")
        msg_id = msg.get("id", 0)

        # Parse reply context
        reply_to = msg.get("reply_to_message_id")

        messages.append({
            "text": text,
            "sender": sender,
            "date": date_str,
            "message_id": msg_id,
            "reply_to_message_id": reply_to,
            "chat_name": chat_name,
        })

    return messages


def create_collection(client: weaviate.WeaviateClient):
    """Create the Weaviate collection if it doesn't exist."""
    if client.collections.exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting and recreating.")
        client.collections.delete(COLLECTION_NAME)

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
        ),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="sender", data_type=DataType.TEXT),
            Property(name="date", data_type=DataType.TEXT),
            Property(name="message_id", data_type=DataType.INT),
            Property(name="reply_to_message_id", data_type=DataType.INT),
            Property(name="chat_name", data_type=DataType.TEXT),
        ],
    )
    print(f"Created collection '{COLLECTION_NAME}'.")


def ingest_messages(
    client: weaviate.WeaviateClient,
    messages: list[dict],
    embedder: EmbeddingModel,
):
    """Batch-insert messages with their embeddings into Weaviate."""
    collection = client.collections.get(COLLECTION_NAME)

    total = len(messages)
    for i in range(0, total, BATCH_SIZE):
        batch = messages[i : i + BATCH_SIZE]
        texts = [m["text"] for m in batch]
        vectors = embedder.embed_documents(texts)

        with collection.batch.dynamic() as weaviate_batch:
            for msg, vec in zip(batch, vectors):
                props = {
                    "text": msg["text"],
                    "sender": msg["sender"],
                    "date": msg["date"],
                    "message_id": msg["message_id"],
                    "chat_name": msg["chat_name"],
                }
                if msg["reply_to_message_id"] is not None:
                    props["reply_to_message_id"] = msg["reply_to_message_id"]

                weaviate_batch.add_object(properties=props, vector=vec)

        done = min(i + BATCH_SIZE, total)
        print(f"Ingested {done}/{total} messages")

    print("Ingestion complete.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path-to-result.json>")
        sys.exit(1)

    json_path = sys.argv[1]
    if not Path(json_path).exists():
        print(f"File not found: {json_path}")
        sys.exit(1)

    print(f"Parsing Telegram export: {json_path}")
    messages = parse_telegram_export(json_path)
    print(f"Found {len(messages)} text messages.")

    if not messages:
        print("No messages to ingest.")
        sys.exit(0)

    print("Loading embedding model (this may take a moment on first run)...")
    embedder = EmbeddingModel()

    print(f"Connecting to Weaviate at {WEAVIATE_URL}")
    client = weaviate.connect_to_local()

    try:
        create_collection(client)
        ingest_messages(client, messages, embedder)
    finally:
        client.close()


if __name__ == "__main__":
    main()
