"""Async OpenRouter client, with per-call token and cost accounting.

Async on purpose: a request spends ~5 seconds waiting on OpenRouter, and under
concurrent users that wait must cost a coroutine, not an OS thread. The client
and the FastAPI handlers were converted together -- `async def` around a blocking
call is strictly worse than staying sync, because it parks the whole event loop.

Every call appends one JSON object to the ledger (`logs/llm_calls.jsonl`):
model, role, token counts, cost, latency. Report over it with
`research/llm_cost_report.py`.

Cost is NOT computed locally. Sending `usage: {"include": true}` makes OpenRouter
return the authoritative `usage.cost` for the call, which stays correct when
prices change or when a fallback model actually served the request. A local
pricing table would silently drift.
"""

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from config import openrouter_key, settings

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_SECONDS = 120
APP_TITLE = "infocom-rag"

# One client for the process: an AsyncClient owns a connection pool, so reusing
# it avoids a fresh TCP + TLS handshake on every call.
_client: httpx.AsyncClient | None = None

# Serialises ledger appends. Without it, concurrent writers each hold their own
# file position and silently overwrite one another -- measured at ~10% of rows
# lost with 8 threads. Correct within one process; multiple uvicorn workers would
# still race, at which point the database becomes the source of truth.
_LEDGER_LOCK = threading.Lock()


def get_client() -> httpx.AsyncClient:
    """Return the shared client, creating it on first use.

    Self-healing if it was closed, so scripts that run their own event loop still
    work. The API closes it explicitly in its lifespan.
    """
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=TIMEOUT_SECONDS)
    return _client


async def close_client() -> None:
    """Close the shared client. Called from the FastAPI lifespan shutdown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class LLMResponse:
    content: str
    usage: TokenUsage
    model: str
    latency_ms: int
    role: str
    raw_usage: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    """Raised when OpenRouter returns an error or an unusable response."""


def _extract_usage(usage: dict[str, Any]) -> TokenUsage:
    """Pull normalized token counts out of OpenRouter's usage object.

    Two of these are NESTED and reading them at the top level silently yields 0:
      - reasoning: usage.completion_tokens_details.reasoning_tokens
      - cache reads: usage.prompt_tokens_details.cached_tokens
    """
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return TokenUsage(
        input_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=int(usage.get("completion_tokens", 0)),
        reasoning_tokens=int(completion_details.get("reasoning_tokens", 0)),
        cached_tokens=int(prompt_details.get("cached_tokens", 0)),
        cost_usd=float(usage.get("cost", 0.0)),
    )


def _build_payload(messages: list[dict[str, str]], model: str,
                   temperature: float | None) -> dict[str, Any]:
    gen = settings.generation
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "usage": {"include": True},        # makes OpenRouter return usage.cost
        "max_tokens": gen.max_output_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
        payload["top_p"] = gen.top_p
        payload["top_k"] = gen.top_k

    chain = [model, *(m for m in gen.fallback if m != model)]
    if len(chain) > 1:
        payload["models"] = chain

    provider: dict[str, Any] = {}
    if gen.pin_provider:
        # Preference, not a hard pin: failover still works on a real outage.
        provider["order"] = [gen.pin_provider]
        provider["allow_fallbacks"] = True
    if provider:
        payload["provider"] = provider
    return payload


def _record(response: LLMResponse) -> None:
    """Append one line to the ledger. Never raises -- accounting must not break a call."""
    path = Path(settings.logging.llm_ledger)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "role": response.role,
        "model": response.model,
        "latency_ms": response.latency_ms,
        **asdict(response.usage),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # The lock is the whole point: see _LEDGER_LOCK above.
        with _LEDGER_LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("Could not write LLM ledger to %s: %s", path, exc)


async def call(messages: list[dict[str, str]], role: str = "answer",
               model: str | None = None, temperature: float | None = None) -> LLMResponse:
    """POST to OpenRouter and return content plus usage.

    `role` is local metadata only (it labels ledger rows, e.g. "answer",
    "eval_judge"); it is never sent to the API.
    """
    gen = settings.generation
    model = model or gen.model
    temperature = gen.temperature if temperature is None else temperature

    payload = _build_payload(messages, model, temperature)
    headers = {
        "Authorization": f"Bearer {openrouter_key()}",
        "X-Title": APP_TITLE,
    }

    start = time.monotonic_ns()
    try:
        resp = await get_client().post(OPENROUTER_URL, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise LLMError(f"OpenRouter request failed: {exc}") from exc
    latency_ms = (time.monotonic_ns() - start) // 1_000_000

    if resp.status_code != 200:
        raise LLMError(f"OpenRouter returned HTTP {resp.status_code}: {resp.text[:400]}")

    data = resp.json()
    # OpenRouter can answer 200 with an error body (e.g. upstream provider error).
    if "error" in data and "choices" not in data:
        raise LLMError(f"OpenRouter error: {data['error']}")
    if not data.get("choices"):
        raise LLMError(f"OpenRouter returned no choices: {json.dumps(data)[:400]}")

    message = data["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not content:
        finish = data["choices"][0].get("finish_reason")
        raise LLMError(f"OpenRouter returned empty content (finish_reason={finish})")

    raw_usage = data.get("usage", {})
    usage = _extract_usage(raw_usage)
    served_by = data.get("model", model)

    response = LLMResponse(
        content=content, usage=usage, model=served_by,
        latency_ms=latency_ms, role=role, raw_usage=raw_usage,
    )

    logger.info(
        "LLM %s model=%s in=%d out=%d cost=$%.6f %dms",
        role, served_by, usage.input_tokens, usage.output_tokens,
        usage.cost_usd, latency_ms,
    )
    if served_by != model:
        logger.warning("Fallback served this call: asked %s, got %s", model, served_by)

    _record(response)
    return response


async def answer(question: str, context: str, role: str = "answer") -> LLMResponse:
    """Run the pinned RAG system prompt over `context` and `question`."""
    messages = [
        {"role": "system", "content": settings.system_prompt},
        {"role": "user", "content": f"Excerpts:\n---\n{context}\n---\n\nQuestion: {question}"},
    ]
    return await call(messages, role=role)
