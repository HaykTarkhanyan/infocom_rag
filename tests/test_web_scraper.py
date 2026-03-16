"""Tests for web_scraper.py — HTML conversion and article processing (no network)."""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

# Mock requests before importing the scraper
sys.modules.setdefault("requests", MagicMock())

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from web_scraper import html_to_text


class TestHtmlToText:
    def test_plain_text(self):
        assert html_to_text("Hello world") == "Hello world"

    def test_strips_tags(self):
        assert html_to_text("<p>Hello <b>world</b></p>") == "Hello world"

    def test_preserves_paragraph_breaks(self):
        result = html_to_text("<p>First paragraph</p><p>Second paragraph</p>")
        assert "First paragraph" in result
        assert "Second paragraph" in result
        assert "\n\n" in result

    def test_br_to_newline(self):
        result = html_to_text("Line one<br>Line two<br/>Line three")
        assert "Line one\nLine two\nLine three" == result

    def test_decodes_html_entities(self):
        assert html_to_text("&amp; &lt; &gt; &quot;") == "& < > \""

    def test_collapses_whitespace(self):
        result = html_to_text("<p>  lots   of    spaces  </p>")
        assert "lots of spaces" in result

    def test_empty_string(self):
        assert html_to_text("") == ""

    def test_nested_tags(self):
        result = html_to_text("<div><p>Hello <a href='#'>link <span>text</span></a></p></div>")
        assert "Hello link text" in result

    def test_armenian_text(self):
        result = html_to_text("<p>Բարեdelays Ձdelays!</p>")
        assert "Բարեdelays Ձdelays!" in result
