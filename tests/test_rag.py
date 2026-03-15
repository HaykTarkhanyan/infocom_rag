"""Tests for rag.py — unit tests with mocked external services."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

# Mock heavy dependencies
sys.modules.setdefault("weaviate", MagicMock())
sys.modules.setdefault("weaviate.classes", MagicMock())
sys.modules.setdefault("weaviate.classes.config", MagicMock())
sys.modules.setdefault("google", MagicMock())
sys.modules.setdefault("google.genai", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.nn", MagicMock())
sys.modules.setdefault("torch.nn.functional", MagicMock())
sys.modules.setdefault("transformers", MagicMock())


class TestRAGFormatContext:
    @patch("rag.genai")
    @patch("rag.weaviate")
    @patch("rag.EmbeddingModel")
    def test_format_context_joins_texts(self, mock_emb, mock_weaviate, mock_genai):
        mock_client = MagicMock()
        mock_weaviate.connect_to_local.return_value = mock_client
        mock_genai.Client.return_value = MagicMock()

        from rag import RAG
        rag = RAG()

        results = [
            {"text": "Line 1", "chunk_type": "single_message", "distance": 0.1},
            {"text": "Line 2", "chunk_type": "sliding_window", "distance": 0.2},
        ]
        ctx = rag.format_context(results)
        assert ctx == "Line 1\nLine 2"
        rag.close()

    @patch("rag.genai")
    @patch("rag.weaviate")
    @patch("rag.EmbeddingModel")
    def test_format_context_empty(self, mock_emb, mock_weaviate, mock_genai):
        mock_client = MagicMock()
        mock_weaviate.connect_to_local.return_value = mock_client
        mock_genai.Client.return_value = MagicMock()

        from rag import RAG
        rag = RAG()
        assert rag.format_context([]) == ""
        rag.close()

    @patch("rag.genai")
    @patch("rag.weaviate")
    @patch("rag.EmbeddingModel")
    def test_answer_no_results(self, mock_emb, mock_weaviate, mock_genai):
        mock_client = MagicMock()
        mock_weaviate.connect_to_local.return_value = mock_client
        mock_genai.Client.return_value = MagicMock()

        from rag import RAG
        rag = RAG()
        # Mock retrieve to return empty
        rag.retrieve = MagicMock(return_value=[])
        answer, sources = rag.answer("test question")
        assert "couldn't find" in answer.lower()
        assert sources == []
        rag.close()

    @patch("rag.genai")
    @patch("rag.weaviate")
    @patch("rag.EmbeddingModel")
    def test_settings_are_configurable(self, mock_emb, mock_weaviate, mock_genai):
        mock_client = MagicMock()
        mock_weaviate.connect_to_local.return_value = mock_client
        mock_genai.Client.return_value = MagicMock()

        from rag import RAG
        rag = RAG(top_k=5, max_distance=0.5, temperature=0.7)
        assert rag.top_k == 5
        assert rag.max_distance == 0.5
        assert rag.temperature == 0.7
        rag.close()
