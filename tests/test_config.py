"""Tests for config.py — verify defaults and type coercions."""

import os

# Set required env vars before importing config
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

import config


class TestConfigDefaults:
    def test_weaviate_url_default(self):
        assert "localhost" in config.WEAVIATE_URL or "127.0.0.1" in config.WEAVIATE_URL

    def test_embedding_dim(self):
        assert config.EMBEDDING_DIM == 768

    def test_max_tokens(self):
        assert config.MAX_TOKENS == 512

    def test_collection_name_is_string(self):
        assert isinstance(config.COLLECTION_NAME, str)
        assert len(config.COLLECTION_NAME) > 0

    def test_chunking_strategy_valid(self):
        valid = {"single_message", "sliding_window", "conversation_thread"}
        assert config.CHUNKING_STRATEGY in valid

    def test_numeric_configs_are_ints(self):
        assert isinstance(config.CHUNK_WINDOW_SIZE, int)
        assert isinstance(config.CHUNK_OVERLAP, int)
        assert isinstance(config.THREAD_MAX_MESSAGES, int)
        assert isinstance(config.MIN_MESSAGE_LENGTH, int)
        assert isinstance(config.INGEST_BATCH_SIZE, int)
        assert isinstance(config.TOP_K, int)

    def test_numeric_configs_positive(self):
        assert config.CHUNK_WINDOW_SIZE > 0
        assert config.CHUNK_OVERLAP >= 0
        assert config.THREAD_MAX_MESSAGES > 0
        assert config.INGEST_BATCH_SIZE > 0
        assert config.TOP_K > 0

    def test_distance_metric_valid(self):
        assert config.DISTANCE_METRIC in {"cosine", "dot", "l2"}

    def test_gemini_temperature_range(self):
        assert 0.0 <= config.GEMINI_TEMPERATURE <= 2.0

    def test_max_distance_range(self):
        assert 0.0 <= config.MAX_DISTANCE <= 2.0

    def test_system_prompt_not_empty(self):
        assert isinstance(config.SYSTEM_PROMPT, str)
        assert len(config.SYSTEM_PROMPT) > 10
