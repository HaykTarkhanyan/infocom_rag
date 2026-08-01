"""Tests for ingest.py — parsing and chunking logic."""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock

import pytest

# Ensure config module doesn't crash when .env is missing
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

# Mock heavy dependencies that aren't needed for pure-logic tests
sys.modules.setdefault("weaviate", MagicMock())
sys.modules.setdefault("weaviate.classes", MagicMock())
sys.modules.setdefault("weaviate.classes.config", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.nn", MagicMock())
sys.modules.setdefault("torch.nn.functional", MagicMock())
sys.modules.setdefault("transformers", MagicMock())

from ingest import (
    extract_text,
    parse_telegram_export,
    parse_web_articles,
    detect_source,
    chunk_single_message,
    chunk_sliding_window,
    chunk_conversation_thread,
    apply_chunking,
)


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_plain_string(self):
        assert extract_text("Hello world") == "Hello world"

    def test_empty_string(self):
        assert extract_text("") == ""

    def test_list_of_strings(self):
        assert extract_text(["Hello", " ", "world"]) == "Hello world"

    def test_list_with_dicts(self):
        parts = [
            "Check ",
            {"type": "link", "text": "this link"},
            " out",
        ]
        assert extract_text(parts) == "Check this link out"

    def test_list_with_dict_missing_text_raises(self):
        parts = [{"type": "bold"}, "hello"]
        with pytest.raises(KeyError, match="no 'text' key"):
            extract_text(parts)

    def test_non_string_non_list_raises(self):
        with pytest.raises(TypeError, match="Expected str, list, or None"):
            extract_text(123)

    def test_none_returns_empty(self):
        assert extract_text(None) == ""

    def test_mixed_types_in_list(self):
        parts = ["prefix ", {"text": "middle"}, " suffix"]
        assert extract_text(parts) == "prefix middle suffix"


# ---------------------------------------------------------------------------
# parse_telegram_export
# ---------------------------------------------------------------------------


def _make_export(messages, chat_name="TestChat"):
    """Helper: write a Telegram export JSON and return its path."""
    data = {"name": chat_name, "messages": messages}
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(data, tmp, ensure_ascii=False)
    tmp.close()
    return tmp.name


class TestParseTelegramExport:
    def test_basic_messages(self):
        path = _make_export([
            {"id": 1, "type": "message", "text": "Hello", "from": "Alice", "date": "2024-01-01T12:00:00"},
            {"id": 2, "type": "message", "text": "Hi there", "from": "Bob", "date": "2024-01-01T12:01:00"},
        ])
        try:
            msgs = parse_telegram_export(path)
            assert len(msgs) == 2
            assert msgs[0]["text"] == "Hello"
            assert msgs[0]["sender"] == "Alice"
            assert msgs[1]["text"] == "Hi there"
        finally:
            os.unlink(path)

    def test_skips_non_message_types(self):
        path = _make_export([
            {"id": 1, "type": "service", "text": "joined", "from": "Alice", "date": "2024-01-01"},
            {"id": 2, "type": "message", "text": "Hello", "from": "Bob", "date": "2024-01-01"},
        ])
        try:
            msgs = parse_telegram_export(path)
            assert len(msgs) == 1
            assert msgs[0]["sender"] == "Bob"
        finally:
            os.unlink(path)

    def test_skips_empty_text(self):
        path = _make_export([
            {"id": 1, "type": "message", "text": "", "from": "Alice", "date": "2024-01-01"},
            {"id": 2, "type": "message", "text": "Real message", "from": "Bob", "date": "2024-01-01"},
        ])
        try:
            msgs = parse_telegram_export(path)
            assert len(msgs) == 1
        finally:
            os.unlink(path)

    def test_reply_to_message_id(self):
        path = _make_export([
            {"id": 1, "type": "message", "text": "First", "from": "Alice", "date": "2024-01-01"},
            {"id": 2, "type": "message", "text": "Reply", "from": "Bob", "date": "2024-01-01", "reply_to_message_id": 1},
        ])
        try:
            msgs = parse_telegram_export(path)
            assert msgs[1]["reply_to_message_id"] == 1
            assert msgs[0]["reply_to_message_id"] is None
        finally:
            os.unlink(path)

    def test_chat_name_preserved(self):
        path = _make_export(
            [{"id": 1, "type": "message", "text": "Hi", "from": "X", "date": "2024-01-01"}],
            chat_name="MyGroup",
        )
        try:
            msgs = parse_telegram_export(path)
            assert msgs[0]["chat_name"] == "MyGroup"
        finally:
            os.unlink(path)

    def test_actor_fallback(self):
        """When 'from' is missing, falls back to 'actor'."""
        path = _make_export([
            {"id": 1, "type": "message", "text": "Hello", "actor": "Charlie", "date": "2024-01-01"},
        ])
        try:
            msgs = parse_telegram_export(path)
            assert msgs[0]["sender"] == "Charlie"
        finally:
            os.unlink(path)

    def test_complex_text_field(self):
        """Text field with mixed string/dict list."""
        path = _make_export([
            {
                "id": 1,
                "type": "message",
                "text": ["Hello ", {"type": "bold", "text": "world"}],
                "from": "Alice",
                "date": "2024-01-01",
            },
        ])
        try:
            msgs = parse_telegram_export(path)
            assert msgs[0]["text"] == "Hello world"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------


def _sample_messages(n=10):
    """Create N sample parsed messages."""
    return [
        {
            "text": f"Message {i}",
            "sender": f"User{i % 3}",
            "date": f"2024-01-01T12:{i:02d}:00",
            "message_id": i,
            "reply_to_message_id": None,
            "chat_name": "TestChat",
        }
        for i in range(1, n + 1)
    ]


class TestChunkSingleMessage:
    def test_one_chunk_per_message(self):
        msgs = _sample_messages(5)
        chunks = chunk_single_message(msgs)
        assert len(chunks) == 5

    def test_chunk_contains_metadata(self):
        msgs = _sample_messages(1)
        chunks = chunk_single_message(msgs)
        c = chunks[0]
        assert c["chunk_type"] == "single_message"
        assert c["sender"] == "User1"
        assert "Message 1" in c["text"]

    def test_empty_input(self):
        assert chunk_single_message([]) == []


class TestChunkSlidingWindow:
    def test_window_size(self):
        msgs = _sample_messages(10)
        chunks = chunk_sliding_window(msgs, window_size=3, overlap=1)
        # Each chunk should combine multiple messages
        assert all("\n" in c["text"] for c in chunks if len(chunks) > 1)

    def test_overlap_produces_more_chunks(self):
        msgs = _sample_messages(10)
        no_overlap = chunk_sliding_window(msgs, window_size=5, overlap=0)
        with_overlap = chunk_sliding_window(msgs, window_size=5, overlap=3)
        assert len(with_overlap) >= len(no_overlap)

    def test_single_message_window(self):
        msgs = _sample_messages(3)
        chunks = chunk_sliding_window(msgs, window_size=1, overlap=0)
        assert len(chunks) == 3

    def test_window_larger_than_data(self):
        msgs = _sample_messages(3)
        chunks = chunk_sliding_window(msgs, window_size=10, overlap=0)
        assert len(chunks) == 1
        # Should contain all 3 messages
        for m in msgs:
            assert m["text"] in chunks[0]["text"]

    def test_empty_input(self):
        assert chunk_sliding_window([], window_size=5, overlap=2) == []

    def test_chunk_type_label(self):
        msgs = _sample_messages(5)
        chunks = chunk_sliding_window(msgs, window_size=3, overlap=1)
        assert chunks[0]["chunk_type"] == "sliding_window_3_1"


class TestChunkConversationThread:
    def test_flat_messages_no_replies(self):
        msgs = _sample_messages(5)
        chunks = chunk_conversation_thread(msgs, max_per_thread=100)
        # Each message is its own thread root
        assert len(chunks) == 5

    def test_reply_chain_grouped(self):
        msgs = [
            {"text": "Root", "sender": "A", "date": "2024-01-01", "message_id": 1, "reply_to_message_id": None, "chat_name": "Test"},
            {"text": "Reply1", "sender": "B", "date": "2024-01-01", "message_id": 2, "reply_to_message_id": 1, "chat_name": "Test"},
            {"text": "Reply2", "sender": "C", "date": "2024-01-01", "message_id": 3, "reply_to_message_id": 1, "chat_name": "Test"},
            {"text": "Standalone", "sender": "D", "date": "2024-01-01", "message_id": 4, "reply_to_message_id": None, "chat_name": "Test"},
        ]
        chunks = chunk_conversation_thread(msgs, max_per_thread=100)
        # 2 chunks: one thread (msgs 1,2,3) and one standalone (msg 4)
        assert len(chunks) == 2
        thread_chunk = chunks[0]
        assert "Root" in thread_chunk["text"]
        assert "Reply1" in thread_chunk["text"]
        assert "Reply2" in thread_chunk["text"]

    def test_long_thread_split(self):
        msgs = [
            {"text": "Root", "sender": "A", "date": "2024-01-01", "message_id": 1, "reply_to_message_id": None, "chat_name": "Test"},
        ]
        for i in range(2, 12):
            msgs.append({
                "text": f"Reply{i}",
                "sender": "B",
                "date": "2024-01-01",
                "message_id": i,
                "reply_to_message_id": 1,
                "chat_name": "Test",
            })
        chunks = chunk_conversation_thread(msgs, max_per_thread=5)
        # 11 messages in one thread, split into ceil(11/5) = 3 chunks
        assert len(chunks) == 3

    def test_empty_input(self):
        assert chunk_conversation_thread([], max_per_thread=10) == []

    def test_chunk_type(self):
        msgs = _sample_messages(2)
        chunks = chunk_conversation_thread(msgs, max_per_thread=100)
        assert all(c["chunk_type"] == "conversation_thread" for c in chunks)


class TestApplyChunking:
    def test_known_strategies(self):
        msgs = _sample_messages(5)
        for strategy in ("single_message", "sliding_window", "conversation_thread"):
            chunks = apply_chunking(msgs, strategy)
            assert len(chunks) > 0

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            apply_chunking(_sample_messages(3), "nonexistent")


# ---------------------------------------------------------------------------
# Web article parsing
# ---------------------------------------------------------------------------


def _write_jsonl(articles: list[dict]) -> str:
    """Write articles as JSONL and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for a in articles:
        tmp.write(json.dumps(a, ensure_ascii=False) + "\n")
    tmp.close()
    return tmp.name


class TestParseWebArticles:
    def test_basic_articles(self):
        path = _write_jsonl([
            {"id": 1, "title": "Title One", "content": "Body one", "author": "Alice", "date": "2026-01-01T12:00:00", "url": "https://example.com/1", "categories": ["news"]},
            {"id": 2, "title": "Title Two", "content": "Body two", "author": "Bob", "date": "2026-01-02T12:00:00", "url": "https://example.com/2", "categories": ["indepth"]},
        ])
        try:
            msgs = parse_web_articles(path)
            assert len(msgs) == 2
            assert "Title One" in msgs[0]["text"]
            assert "Body one" in msgs[0]["text"]
            assert msgs[0]["sender"] == "Alice"
            assert msgs[0]["chat_name"] == "news"
            assert msgs[1]["chat_name"] == "indepth"
        finally:
            os.unlink(path)

    def test_empty_content_skipped(self):
        path = _write_jsonl([
            {"id": 1, "title": "", "content": "", "author": "X", "date": "", "url": "", "categories": []},
            {"id": 2, "title": "Real", "content": "Article", "author": "Y", "date": "", "url": "", "categories": []},
        ])
        try:
            msgs = parse_web_articles(path)
            assert len(msgs) == 1
            assert msgs[0]["message_id"] == 2
        finally:
            os.unlink(path)

    def test_title_only_article(self):
        path = _write_jsonl([
            {"id": 1, "title": "Just a title", "content": "", "author": "A", "date": "", "url": "", "categories": []},
        ])
        try:
            msgs = parse_web_articles(path)
            assert len(msgs) == 1
            assert msgs[0]["text"] == "Just a title"
        finally:
            os.unlink(path)

    def test_no_categories_defaults(self):
        path = _write_jsonl([
            {"id": 1, "title": "T", "content": "C", "author": "A", "date": "", "url": "", "categories": []},
        ])
        try:
            msgs = parse_web_articles(path)
            assert msgs[0]["chat_name"] == "infocom.am"
        finally:
            os.unlink(path)

    def test_multiple_categories_joined(self):
        path = _write_jsonl([
            {"id": 1, "title": "T", "content": "C", "author": "A", "date": "", "url": "", "categories": ["news", "indepth"]},
        ])
        try:
            msgs = parse_web_articles(path)
            assert msgs[0]["chat_name"] == "news, indepth"
        finally:
            os.unlink(path)

    def test_reply_to_always_none(self):
        path = _write_jsonl([
            {"id": 1, "title": "T", "content": "C", "author": "A", "date": "", "url": "", "categories": []},
        ])
        try:
            msgs = parse_web_articles(path)
            assert msgs[0]["reply_to_message_id"] is None
        finally:
            os.unlink(path)

    def test_blank_lines_skipped(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        tmp.write('{"id":1,"title":"T","content":"C","author":"A","date":"","url":"","categories":[]}\n')
        tmp.write("\n")
        tmp.write('{"id":2,"title":"T2","content":"C2","author":"B","date":"","url":"","categories":[]}\n')
        tmp.close()
        try:
            msgs = parse_web_articles(tmp.name)
            assert len(msgs) == 2
        finally:
            os.unlink(tmp.name)


class TestDetectSource:
    def test_detects_web_jsonl(self):
        path = _write_jsonl([
            {"id": 1, "title": "T", "content": "C", "author": "A", "date": "", "url": "", "categories": []},
        ])
        try:
            assert detect_source(path) == "web"
        finally:
            os.unlink(path)

    def test_detects_telegram_export(self):
        path = _make_export([
            {"id": 1, "type": "message", "text": "Hello", "from": "Alice", "date": "2024-01-01"},
        ])
        try:
            assert detect_source(path) == "telegram"
        finally:
            os.unlink(path)
