"""Tests for embeddings.py — unit tests using mocking (no GPU/model download needed)."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

# Ensure torch is available (real import needed for tensor ops in tests)
try:
    import torch
except ImportError:
    pytest.skip("torch not installed", allow_module_level=True)


class TestEmbeddingModel:
    @patch("embeddings.AutoModel")
    @patch("embeddings.AutoTokenizer")
    def test_init_loads_model(self, mock_tokenizer_cls, mock_model_cls):
        from embeddings import EmbeddingModel

        model = EmbeddingModel("fake-model")
        mock_tokenizer_cls.from_pretrained.assert_called_once_with("fake-model")
        mock_model_cls.from_pretrained.assert_called_once_with("fake-model")

    @patch("embeddings.AutoModel")
    @patch("embeddings.AutoTokenizer")
    def test_embed_query_returns_single_vector(self, mock_tokenizer_cls, mock_model_cls):
        import torch
        from embeddings import EmbeddingModel

        # Mock the model output
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        hidden = torch.randn(1, 5, 768)
        mock_model.return_value = MagicMock(last_hidden_state=hidden)

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }

        emb = EmbeddingModel("fake-model")
        result = emb.embed_query("test query")
        assert isinstance(result, list)
        assert len(result) == 768

    @patch("embeddings.AutoModel")
    @patch("embeddings.AutoTokenizer")
    def test_embed_documents_returns_list_of_vectors(self, mock_tokenizer_cls, mock_model_cls):
        import torch
        from embeddings import EmbeddingModel

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        hidden = torch.randn(3, 5, 768)
        mock_model.return_value = MagicMock(last_hidden_state=hidden)

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(3, 5, dtype=torch.long),
            "attention_mask": torch.ones(3, 5, dtype=torch.long),
        }

        emb = EmbeddingModel("fake-model")
        results = emb.embed_documents(["a", "b", "c"])
        assert len(results) == 3
        assert all(len(v) == 768 for v in results)

    @patch("embeddings.AutoModel")
    @patch("embeddings.AutoTokenizer")
    def test_embed_adds_prefix(self, mock_tokenizer_cls, mock_model_cls):
        import torch
        from embeddings import EmbeddingModel

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        hidden = torch.randn(1, 5, 768)
        mock_model.return_value = MagicMock(last_hidden_state=hidden)

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }

        emb = EmbeddingModel("fake-model")
        emb.embed(["hello"], prefix="query")
        call_args = mock_tokenizer.call_args[0][0]
        assert call_args == ["query: hello"]
