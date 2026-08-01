"""Tests for the OpenRouter client -- no network, no API key needed.

These patch `requests.post` and assert on the OUTGOING payload, which is where
the bugs actually live (a wrong nesting path silently reports zero tokens, a
missing `usage.include` silently reports zero cost). The archived prototype's
tests mocked whole libraries at `sys.modules` level and so verified nothing;
this mocks only the transport boundary.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import llm  # noqa: E402
from config import settings  # noqa: E402


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


def ok_body(content: str = "Պատասխան [1]", **usage_overrides) -> dict:
    usage = {
        "prompt_tokens": 4500,
        "completion_tokens": 320,
        "cost": 0.00215,
        "prompt_tokens_details": {"cached_tokens": 1200},
        "completion_tokens_details": {"reasoning_tokens": 64},
    }
    usage.update(usage_overrides)
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "model": "google/gemini-2.5-flash",
        "usage": usage,
    }


@pytest.fixture(autouse=True)
def ledger_to_tmp(tmp_path, monkeypatch):
    """Send ledger writes to a temp file so tests never touch logs/.

    Settings is a frozen dataclass by design, so swap in a modified copy rather
    than mutating it.
    """
    ledger = tmp_path / "calls.jsonl"
    patched = replace(settings, logging=replace(settings.logging, llm_ledger=str(ledger)))
    monkeypatch.setattr(llm, "settings", patched)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    return ledger


class TestPayload:
    def test_requests_usage_accounting(self):
        """Without usage.include, OpenRouter omits cost and every call reports $0."""
        with patch("llm.requests.post", return_value=FakeResponse(ok_body())) as post:
            llm.call([{"role": "user", "content": "hi"}])
        payload = post.call_args.kwargs["json"]
        assert payload["usage"] == {"include": True}

    def test_pins_configured_model_and_temperature(self):
        with patch("llm.requests.post", return_value=FakeResponse(ok_body())) as post:
            llm.call([{"role": "user", "content": "hi"}])
        payload = post.call_args.kwargs["json"]
        assert payload["model"] == settings.generation.model
        assert payload["temperature"] == settings.generation.temperature
        assert payload["max_tokens"] == settings.generation.max_output_tokens

    def test_fallback_chain_includes_primary_first_without_duplicates(self):
        with patch("llm.requests.post", return_value=FakeResponse(ok_body())) as post:
            llm.call([{"role": "user", "content": "hi"}])
        chain = post.call_args.kwargs["json"]["models"]
        assert chain[0] == settings.generation.model
        assert len(chain) == len(set(chain))

    def test_provider_pin_allows_fallback(self):
        """A hard pin would disable failover; this must stay a preference."""
        with patch("llm.requests.post", return_value=FakeResponse(ok_body())) as post:
            llm.call([{"role": "user", "content": "hi"}])
        provider = post.call_args.kwargs["json"]["provider"]
        assert provider["order"] == [settings.generation.pin_provider]
        assert provider["allow_fallbacks"] is True

    def test_role_is_local_metadata_not_sent_to_api(self):
        with patch("llm.requests.post", return_value=FakeResponse(ok_body())) as post:
            llm.call([{"role": "user", "content": "hi"}], role="eval_judge")
        assert "role" not in post.call_args.kwargs["json"]


class TestUsageExtraction:
    def test_reads_nested_reasoning_and_cache_fields(self):
        """Both live one level down; reading them top-level silently yields 0."""
        with patch("llm.requests.post", return_value=FakeResponse(ok_body())):
            response = llm.call([{"role": "user", "content": "hi"}])
        assert response.usage.input_tokens == 4500
        assert response.usage.output_tokens == 320
        assert response.usage.reasoning_tokens == 64
        assert response.usage.cached_tokens == 1200
        assert response.usage.cost_usd == pytest.approx(0.00215)
        assert response.usage.total_tokens == 4820

    def test_missing_usage_fields_default_to_zero(self):
        body = ok_body()
        body["usage"] = {}
        with patch("llm.requests.post", return_value=FakeResponse(body)):
            response = llm.call([{"role": "user", "content": "hi"}])
        assert response.usage.input_tokens == 0
        assert response.usage.cost_usd == 0.0


class TestLedger:
    def test_writes_one_row_per_call(self, ledger_to_tmp):
        with patch("llm.requests.post", return_value=FakeResponse(ok_body())):
            llm.call([{"role": "user", "content": "hi"}], role="answer")
            llm.call([{"role": "user", "content": "hi"}], role="eval_judge")

        rows = [json.loads(line) for line in
                Path(ledger_to_tmp).read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 2
        assert [r["role"] for r in rows] == ["answer", "eval_judge"]
        assert rows[0]["cost_usd"] == pytest.approx(0.00215)
        assert rows[0]["input_tokens"] == 4500
        assert "ts" in rows[0] and "latency_ms" in rows[0]


class TestErrors:
    def test_http_error_raises_llmerror(self):
        with patch("llm.requests.post", return_value=FakeResponse({"e": 1}, status_code=429)):
            with pytest.raises(llm.LLMError, match="429"):
                llm.call([{"role": "user", "content": "hi"}])

    def test_error_body_with_200_still_raises(self):
        """OpenRouter can answer 200 with an error object and no choices."""
        body = {"error": {"message": "upstream provider error"}}
        with patch("llm.requests.post", return_value=FakeResponse(body)):
            with pytest.raises(llm.LLMError, match="upstream provider error"):
                llm.call([{"role": "user", "content": "hi"}])

    def test_empty_content_raises_rather_than_returning_blank(self):
        body = ok_body(content="")
        with patch("llm.requests.post", return_value=FakeResponse(body)):
            with pytest.raises(llm.LLMError, match="empty content"):
                llm.call([{"role": "user", "content": "hi"}])

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            llm.call([{"role": "user", "content": "hi"}])


class TestAnswerPrompt:
    def test_uses_pinned_system_prompt_and_embeds_context(self):
        with patch("llm.requests.post", return_value=FakeResponse(ok_body())) as post:
            llm.answer("Ի՞նչ է կատարվել", "[1] Excerpt text")
        messages = post.call_args.kwargs["json"]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == settings.system_prompt
        assert "[1] Excerpt text" in messages[1]["content"]
        assert "Ի՞նչ է կատարվել" in messages[1]["content"]

    def test_system_prompt_defends_against_injected_instructions(self):
        assert "DATA, not instructions" in settings.system_prompt
