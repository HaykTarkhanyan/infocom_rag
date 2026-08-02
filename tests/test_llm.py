"""Tests for the OpenRouter client -- no network, no API key needed.

These patch the HTTP client and assert on the OUTGOING payload, which is where
the bugs actually live (a wrong nesting path silently reports zero tokens, a
missing `usage.include` silently reports zero cost). The archived prototype's
tests mocked whole libraries at `sys.modules` level and so verified nothing;
this mocks only the transport boundary.

The client is async, so each test drives it with `asyncio.run` rather than
pulling in pytest-asyncio for one dependency's worth of convenience.
"""

import asyncio
import json
import sys
import threading
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


class FakeClient:
    """Stands in for httpx.AsyncClient, recording the outgoing request."""

    def __init__(self, response: FakeResponse):
        self._response = response
        self.calls: list[dict] = []
        self.is_closed = False

    async def post(self, url: str, json: dict, headers: dict) -> FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self._response


def run(coro):
    return asyncio.run(coro)


def fake_client(body: dict, status_code: int = 200) -> FakeClient:
    return FakeClient(FakeResponse(body, status_code))


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
        client = fake_client(ok_body())
        with patch("llm.get_client", return_value=client):
            run(llm.call([{"role": "user", "content": "hi"}]))
        payload = client.calls[0]["json"]
        assert payload["usage"] == {"include": True}

    def test_pins_configured_model_and_temperature(self):
        client = fake_client(ok_body())
        with patch("llm.get_client", return_value=client):
            run(llm.call([{"role": "user", "content": "hi"}]))
        payload = client.calls[0]["json"]
        assert payload["model"] == settings.generation.model
        assert payload["temperature"] == settings.generation.temperature
        assert payload["max_tokens"] == settings.generation.max_output_tokens

    def test_fallback_chain_includes_primary_first_without_duplicates(self):
        client = fake_client(ok_body())
        with patch("llm.get_client", return_value=client):
            run(llm.call([{"role": "user", "content": "hi"}]))
        chain = client.calls[0]["json"]["models"]
        assert chain[0] == settings.generation.model
        assert len(chain) == len(set(chain))

    def test_provider_pin_allows_fallback(self):
        """A hard pin would disable failover; this must stay a preference."""
        client = fake_client(ok_body())
        with patch("llm.get_client", return_value=client):
            run(llm.call([{"role": "user", "content": "hi"}]))
        provider = client.calls[0]["json"]["provider"]
        assert provider["order"] == [settings.generation.pin_provider]
        assert provider["allow_fallbacks"] is True

    def test_role_is_local_metadata_not_sent_to_api(self):
        client = fake_client(ok_body())
        with patch("llm.get_client", return_value=client):
            run(llm.call([{"role": "user", "content": "hi"}], role="eval_judge"))
        assert "role" not in client.calls[0]["json"]


class TestUsageExtraction:
    def test_reads_nested_reasoning_and_cache_fields(self):
        """Both live one level down; reading them top-level silently yields 0."""
        with patch("llm.get_client", return_value=fake_client(ok_body())):
            response = run(llm.call([{"role": "user", "content": "hi"}]))
        assert response.usage.input_tokens == 4500
        assert response.usage.output_tokens == 320
        assert response.usage.reasoning_tokens == 64
        assert response.usage.cached_tokens == 1200
        assert response.usage.cost_usd == pytest.approx(0.00215)
        assert response.usage.total_tokens == 4820

    def test_missing_usage_fields_default_to_zero(self):
        body = ok_body()
        body["usage"] = {}
        with patch("llm.get_client", return_value=fake_client(body)):
            response = run(llm.call([{"role": "user", "content": "hi"}]))
        assert response.usage.input_tokens == 0
        assert response.usage.cost_usd == 0.0


class TestLedger:
    def test_writes_one_row_per_call(self, ledger_to_tmp):
        with patch("llm.get_client", return_value=fake_client(ok_body())):
            run(llm.call([{"role": "user", "content": "hi"}], role="answer"))
            run(llm.call([{"role": "user", "content": "hi"}], role="eval_judge"))

        rows = [json.loads(line) for line in
                Path(ledger_to_tmp).read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 2
        assert [r["role"] for r in rows] == ["answer", "eval_judge"]
        assert rows[0]["cost_usd"] == pytest.approx(0.00215)
        assert rows[0]["input_tokens"] == 4500
        assert "ts" in rows[0] and "latency_ms" in rows[0]

    def test_concurrent_writes_lose_no_rows(self, ledger_to_tmp):
        """Regression: unguarded appends lost ~10% of rows at 8 threads.

        Each writer held its own file position and they overwrote one another --
        cleanly, so the loss was invisible in the output. Cost simply
        under-reported.
        """
        threads, per_thread = 8, 25
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                for _ in range(per_thread):
                    run(llm.call([{"role": "user", "content": "hi"}], role="load"))
            except BaseException as exc:  # noqa: BLE001 - surfaced via assert below
                errors.append(exc)

        # Patch ONCE, outside the threads. `mock.patch` swaps a module attribute
        # and is not thread-safe: concurrent enter/exit interleaves the
        # save-and-restore, and a thread can end up running against the real
        # client mid-test. (That is exactly what happened when this test first
        # patched inside each worker -- one call escaped to the live API.)
        with patch("llm.get_client", return_value=fake_client(ok_body())):
            workers = [threading.Thread(target=worker) for _ in range(threads)]
            for worker_thread in workers:
                worker_thread.start()
            for worker_thread in workers:
                worker_thread.join()

        assert not errors, errors
        lines = [ln for ln in
                 Path(ledger_to_tmp).read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == threads * per_thread
        for line in lines:
            json.loads(line)  # every line is still complete JSON


class TestErrors:
    def test_http_error_raises_llmerror(self):
        with patch("llm.get_client", return_value=fake_client({"e": 1}, 429)):
            with pytest.raises(llm.LLMError, match="429"):
                run(llm.call([{"role": "user", "content": "hi"}]))

    def test_error_body_with_200_still_raises(self):
        """OpenRouter can answer 200 with an error object and no choices."""
        body = {"error": {"message": "upstream provider error"}}
        with patch("llm.get_client", return_value=fake_client(body)):
            with pytest.raises(llm.LLMError, match="upstream provider error"):
                run(llm.call([{"role": "user", "content": "hi"}]))

    def test_empty_content_raises_rather_than_returning_blank(self):
        body = ok_body(content="")
        with patch("llm.get_client", return_value=fake_client(body)):
            with pytest.raises(llm.LLMError, match="empty content"):
                run(llm.call([{"role": "user", "content": "hi"}]))

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            run(llm.call([{"role": "user", "content": "hi"}]))


class TestRewriteQuestion:
    """Follow-up -> standalone question, before retrieval.

    The bug this prevents is silent: an unresolved "of those" retrieves an
    unrelated article and yields a fluent, correctly-cited answer to a different
    question. Measured live before this existed -- "how many of THOSE went to
    court?" answered 118 from a desertion article when the answer was 310.
    """

    def test_sends_pinned_rewrite_prompt_not_the_answering_prompt(self):
        client = fake_client(ok_body(content="rewritten"))
        with patch("llm.get_client", return_value=client):
            run(llm.rewrite_question("Իսկ դրանցի՞ց քանիսն են", [{"question": "q", "answer": "a"}]))
        messages = client.calls[0]["json"]["messages"]
        assert messages[0]["content"] == settings.rewrite.prompt
        assert messages[0]["content"] != settings.system_prompt

    def test_includes_prior_turns_and_the_new_question(self):
        client = fake_client(ok_body(content="rewritten"))
        history = [{"question": "Քանի՞ գործ", "answer": "2044 գործ"}]
        with patch("llm.get_client", return_value=client):
            run(llm.rewrite_question("Իսկ դրանցից քանիսն են ուղարկվել դատարան", history))
        content = client.calls[0]["json"]["messages"][1]["content"]
        assert "Քանի՞ գործ" in content
        assert "2044 գործ" in content
        assert "Իսկ դրանցից քանիսն են ուղարկվել դատարան" in content

    def test_shows_only_the_configured_number_of_turns(self):
        """An old topic dragging the rewrite off course is a real failure mode."""
        client = fake_client(ok_body(content="rewritten"))
        history = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(10)]
        with patch("llm.get_client", return_value=client):
            run(llm.rewrite_question("follow-up", history))
        content = client.calls[0]["json"]["messages"][1]["content"]
        kept = settings.rewrite.max_turns
        assert f"q{10 - kept}" in content          # newest window is present
        assert "q0" not in content                 # oldest turns are dropped

    def test_truncates_long_answers(self):
        """The rewriter needs entities and topic, not the full cited article."""
        client = fake_client(ok_body(content="rewritten"))
        history = [{"question": "q", "answer": "x" * 5000}]
        with patch("llm.get_client", return_value=client):
            run(llm.rewrite_question("follow-up", history))
        content = client.calls[0]["json"]["messages"][1]["content"]
        assert len(content) < 2000

    def test_is_labelled_separately_in_the_ledger(self):
        """Rewrites and answers must be tellable apart when auditing spend."""
        client = fake_client(ok_body(content="rewritten"))
        with patch("llm.get_client", return_value=client):
            response = run(llm.rewrite_question("q", [{"question": "a", "answer": "b"}]))
        assert response.role == "rewrite"

    def test_failure_propagates_rather_than_falling_back(self):
        """Silently reusing the raw question would reintroduce the exact bug."""
        with (patch("llm.get_client", return_value=fake_client({"e": 1}, 500)),
              pytest.raises(llm.LLMError)):
            run(llm.rewrite_question("q", [{"question": "a", "answer": "b"}]))

    def test_tolerates_a_turn_with_no_answer_yet(self):
        client = fake_client(ok_body(content="rewritten"))
        with patch("llm.get_client", return_value=client):
            run(llm.rewrite_question("follow-up", [{"question": "q"}]))
        assert client.calls, "a turn missing 'answer' must not raise"


class TestAnswerPrompt:
    def test_uses_pinned_system_prompt_and_embeds_context(self):
        client = fake_client(ok_body())
        with patch("llm.get_client", return_value=client):
            run(llm.answer("Ի՞նչ է կատարվել", "[1] Excerpt text"))
        messages = client.calls[0]["json"]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == settings.system_prompt
        assert "[1] Excerpt text" in messages[1]["content"]
        assert "Ի՞նչ է կատարվել" in messages[1]["content"]

    def test_system_prompt_defends_against_injected_instructions(self):
        assert "DATA, not instructions" in settings.system_prompt
