"""Tests for scraper.py — conversion logic (no network calls)."""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock

import pytest

# Mock telethon so scraper can be imported without it installed
sys.modules.setdefault("telethon", MagicMock())
sys.modules.setdefault("telethon.tl", MagicMock())
sys.modules.setdefault("telethon.tl.types", MagicMock())

from scraper import to_telegram_export_format


class TestToTelegramExportFormat:
    def _write_scraped(self, messages):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(messages, tmp, ensure_ascii=False)
        tmp.close()
        return tmp.name

    def test_basic_conversion(self):
        scraped = [
            {"id": 1, "date": "2024-01-01T12:00:00", "sender_id": 123, "text": "Hello", "reply_to_msg_id": None, "media_file": None},
            {"id": 2, "date": "2024-01-01T12:01:00", "sender_id": 456, "text": "Hi", "reply_to_msg_id": 1, "media_file": None},
        ]
        src = self._write_scraped(scraped)
        out = tempfile.mktemp(suffix=".json")
        try:
            to_telegram_export_format(src, out)
            with open(out) as f:
                result = json.load(f)
            assert result["name"] == "infocomm"
            assert len(result["messages"]) == 2
            assert result["messages"][0]["type"] == "message"
            assert result["messages"][0]["text"] == "Hello"
            assert result["messages"][1]["reply_to_message_id"] == 1
        finally:
            os.unlink(src)
            if os.path.exists(out):
                os.unlink(out)

    def test_empty_messages(self):
        src = self._write_scraped([])
        out = tempfile.mktemp(suffix=".json")
        try:
            to_telegram_export_format(src, out)
            with open(out) as f:
                result = json.load(f)
            assert result["messages"] == []
        finally:
            os.unlink(src)
            if os.path.exists(out):
                os.unlink(out)

    def test_missing_fields_raises(self):
        scraped = [{"id": 10, "text": "Hey"}]
        src = self._write_scraped(scraped)
        out = tempfile.mktemp(suffix=".json")
        try:
            with pytest.raises(KeyError, match="missing required field"):
                to_telegram_export_format(src, out)
        finally:
            os.unlink(src)
            if os.path.exists(out):
                os.unlink(out)
