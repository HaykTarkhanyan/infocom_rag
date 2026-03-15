"""Fetch messages and images from a Telegram group using Telethon.

Usage:
    python scraper.py                          # fetch from default group (infocomm)
    python scraper.py --group infocomm --limit 1000
    python scraper.py --group infocomm --media-dir ./media --limit 500

Credentials are read from .env:
    TELEGRAM_API_ID=...
    TELEGRAM_API_HASH=...

On first run you will be prompted to log in with your phone number.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.types import (
    MessageMediaPhoto,
    MessageMediaDocument,
)

load_dotenv()

API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
SESSION_NAME = os.getenv("TELEGRAM_SESSION_NAME", "scraper_session")

DEFAULT_GROUP = "infocomm"
DEFAULT_LIMIT = 5000
DEFAULT_MEDIA_DIR = "media"
DEFAULT_OUTPUT = "scraped_messages.json"


async def scrape_group(
    group: str,
    limit: int,
    media_dir: str | None,
    output_path: str,
) -> None:
    """Connect to Telegram and scrape messages + media from *group*."""
    if not API_ID or not API_HASH:
        print(
            "Error: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env\n"
            "Get them at https://my.telegram.org"
        )
        sys.exit(1)

    client = TelegramClient(SESSION_NAME, int(API_ID), API_HASH)
    await client.start()

    print(f"Connected. Fetching messages from @{group} (limit={limit})...")
    entity = await client.get_entity(group)

    if media_dir:
        Path(media_dir).mkdir(parents=True, exist_ok=True)

    messages_data = []
    media_count = 0

    async for message in client.iter_messages(entity, limit=limit):
        record = {
            "id": message.id,
            "date": message.date.isoformat() if message.date else None,
            "sender_id": message.sender_id,
            "text": message.text or "",
            "reply_to_msg_id": (
                message.reply_to.reply_to_msg_id if message.reply_to else None
            ),
            "media_file": None,
        }

        # Download photos and documents
        if media_dir and message.media:
            try:
                if isinstance(message.media, MessageMediaPhoto):
                    filename = f"photo_{message.id}.jpg"
                    filepath = os.path.join(media_dir, filename)
                    await client.download_media(message, file=filepath)
                    record["media_file"] = filename
                    media_count += 1
                elif isinstance(message.media, MessageMediaDocument):
                    mime = getattr(message.media.document, "mime_type", "") or ""
                    if mime.startswith("image/"):
                        ext = mime.split("/")[-1]
                        filename = f"doc_{message.id}.{ext}"
                        filepath = os.path.join(media_dir, filename)
                        await client.download_media(message, file=filepath)
                        record["media_file"] = filename
                        media_count += 1
            except Exception as e:
                print(f"  Warning: failed to download media for msg {message.id}: {e}")

        messages_data.append(record)

    await client.disconnect()

    # Sort chronologically (oldest first)
    messages_data.sort(key=lambda m: m["id"])

    # Write to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(messages_data, f, ensure_ascii=False, indent=2)

    text_count = sum(1 for m in messages_data if m["text"])
    print(f"Done. Scraped {len(messages_data)} messages ({text_count} with text, {media_count} media files).")
    print(f"Saved to {output_path}")
    if media_dir:
        print(f"Media saved to {media_dir}/")


def to_telegram_export_format(scraped_path: str, output_path: str = "result.json") -> None:
    """Convert scraper output to Telegram Desktop export format for ingest.py compatibility.

    This produces a ``result.json`` that ``ingest.py`` can parse directly.
    """
    with open(scraped_path, "r", encoding="utf-8") as f:
        messages = json.load(f)

    converted = {
        "name": "infocomm",
        "type": "public_supergroup",
        "messages": [],
    }

    for m in messages:
        converted["messages"].append({
            "id": m["id"],
            "type": "message",
            "date": m.get("date", ""),
            "from": str(m.get("sender_id", "Unknown")),
            "text": m.get("text", ""),
            "reply_to_message_id": m.get("reply_to_msg_id"),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(messages)} messages to Telegram export format: {output_path}")
    print(f"You can now run: python ingest.py {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape a Telegram group for messages and images")
    parser.add_argument("--group", default=DEFAULT_GROUP, help=f"Group username (default: {DEFAULT_GROUP})")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"Max messages to fetch (default: {DEFAULT_LIMIT})")
    parser.add_argument("--media-dir", default=DEFAULT_MEDIA_DIR, help=f"Directory to save media (default: {DEFAULT_MEDIA_DIR})")
    parser.add_argument("--no-media", action="store_true", help="Skip downloading media files")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    parser.add_argument(
        "--convert", action="store_true",
        help="After scraping, also convert to Telegram Desktop export format for ingest.py",
    )

    args = parser.parse_args()
    media_dir = None if args.no_media else args.media_dir

    asyncio.run(scrape_group(args.group, args.limit, media_dir, args.output))

    if args.convert:
        to_telegram_export_format(args.output)


if __name__ == "__main__":
    main()
