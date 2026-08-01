"""Chainlit UI for the infocom.am RAG.

Calls the FastAPI `/ask` endpoint over HTTP rather than importing the pipeline,
so the UI and the eval harness exercise the same code path. Start both:

    python -m uvicorn api:app --app-dir src --port 8000
    chainlit run chainlit_app.py -w --port 8001

Two `cl.Step`s render the pipeline as a collapsible tree -- retrieval (which
chunks, what scores, which fell below the cut) and generation (assembled prompt,
tokens, cost). That is the debug view; it costs nothing to leave on because
Chainlit collapses steps by default.

Token counts, cost and the pinned config go into `steps.metadata`, which is what
preserves per-answer accounting after adopting Chainlit's schema instead of our
own.
"""

import json
import os
import secrets
import sys
from pathlib import Path

import chainlit as cl
import httpx
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from db import async_dsn

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120


@cl.password_auth_callback
def auth(username: str, password: str) -> cl.User | None:
    """Gate the UI behind a shared password when APP_PASSWORD is set.

    Deployed publicly without this, anyone who finds the URL spends the
    OpenRouter key -- there is no per-user cap or rate limit yet. Chainlit only
    registers this callback if the env var exists, so local development stays
    frictionless.

    Compared with `secrets.compare_digest` rather than `==` so the check does not
    leak the password's length through timing.
    """
    expected = os.environ.get("APP_PASSWORD")
    if not expected:
        return None
    if secrets.compare_digest(password, expected):
        return cl.User(identifier=username or "user")
    return None


@cl.data_layer
def get_data_layer() -> SQLAlchemyDataLayer:
    """Persist threads, steps and feedback to Neon.

    No storage_provider is configured: that is only needed for file-backed
    elements (images, PDFs). Our sources are text rendered inline in the message,
    so nothing needs object storage.
    """
    return SQLAlchemyDataLayer(conninfo=async_dsn())


@cl.on_chat_start
async def on_chat_start() -> None:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            health = (await client.get(f"{API_URL}/health")).json()
    except httpx.HTTPError as exc:
        await cl.Message(
            content=(
                f"**Cannot reach the API at {API_URL}.**\n\n"
                "Start it with:\n"
                "```\npython -m uvicorn api:app --app-dir src --port 8000\n```\n\n"
                f"`{exc}`"
            )
        ).send()
        return

    corpus = health["corpus"]
    await cl.Message(
        content=(
            f"Ask me about **{corpus['articles']} infocom.am articles** "
            f"({corpus['chunks']:,} chunks). Questions in Armenian, English or Russian.\n\n"
            f"- model `{health['model']}` · retriever `{health['retriever']}`"
            + (f" (`{health['embedding_model'].rsplit('/', 1)[-1]}`)"
               if health.get("embedding_model") else "") + "\n"
            "- expand **Retrieval** / **Generation** under any answer to see "
            "which chunks were used, their scores, the assembled prompt and the cost."
        )
    ).send()


def _format_sources(sources: list[dict]) -> str:
    """Render sources grouped by article.

    Several chunks routinely come from the same article, so a flat list repeats
    the same title. Group by post_id and keep every citation number against it,
    because the model cites chunk positions -- renumbering here would break the
    mapping between [n] in the answer and the source it came from.
    """
    by_article: dict[int, dict] = {}
    for src in sources:
        entry = by_article.setdefault(
            src["post_id"],
            {"title": src["title"], "url": src["url"],
             "published": (src.get("published") or "")[:10], "ns": []},
        )
        entry["ns"].append(src["n"])

    lines = ["", "---", "**Աղբյուրներ**"]
    for entry in by_article.values():
        marks = ", ".join(f"[{n}]" for n in entry["ns"])
        lines.append(f"- {marks} [{entry['title']}]({entry['url']}) · {entry['published']}")
    return "\n".join(lines)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    question = message.content.strip()
    if not question:
        return

    async with cl.Step(name="Retrieval", type="retrieval") as retrieval_step:
        retrieval_step.input = question
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{API_URL}/ask", json={"question": question}
                )
        except httpx.HTTPError as exc:
            retrieval_step.is_error = True
            retrieval_step.output = str(exc)
            await cl.Message(content=f"**API unreachable.** `{exc}`").send()
            return

        if response.status_code == 404:
            retrieval_step.output = "no matching chunks"
            await cl.Message(
                content="Ոչինչ չգտնվեց այս հարցման համար։ Փորձեք այլ ձևակերպում։"
            ).send()
            return
        if response.status_code != 200:
            retrieval_step.is_error = True
            retrieval_step.output = response.text[:500]
            await cl.Message(
                content=f"**API error {response.status_code}.**\n```\n{response.text[:400]}\n```"
            ).send()
            return

        data = response.json()
        sources = data["sources"]
        def score_label(src: dict) -> str:
            # BM25 and cosine are unrelated scales, so name which one this is
            # rather than printing a bare number that means different things.
            if src.get("distance") is not None:
                return f"cos {src['score']:.4f} dist {src['distance']:.4f}"
            return f"bm25 {src['score']:>7.2f}"

        retrieval_step.output = "\n".join(
            f"[{s['n']}] {score_label(s)}  {s['n_tokens']:>3} tok  {s['title'][:52]}"
            + (f"\n      § {s['heading'][:56]}" if s.get("heading") else "")
            for s in sources
        ) or "no hits"
        retrieval_step.metadata = {
            "n_sources": len(sources),
            "retrieval_ms": data["retrieval_ms"],
            "chunk_ids": [s["chunk_id"] for s in sources],
            "retriever": data["config"]["retriever"],
            "embedding_model": data["config"].get("embedding_model"),
            "max_distance": data["config"].get("max_distance"),
        }

    usage = data["usage"]
    async with cl.Step(name="Generation", type="llm") as generation_step:
        generation_step.input = data["prompt"]
        generation_step.output = (
            f"model         {data['model']}\n"
            f"input tokens  {usage['input_tokens']:,}\n"
            f"output tokens {usage['output_tokens']:,}\n"
            f"cost          ${usage['cost_usd']:.6f}\n"
            f"latency       {data['generation_ms']:,} ms\n\n"
            f"config: {json.dumps(data['config'], ensure_ascii=False)}"
        )
        # This metadata is the reason per-answer accounting survives using
        # Chainlit's schema rather than our own.
        generation_step.metadata = {
            "model": data["model"],
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "cached_tokens": usage["cached_tokens"],
            "cost_usd": usage["cost_usd"],
            "generation_ms": data["generation_ms"],
            "config": data["config"],
        }

    # Plain markdown, not HTML: Chainlit's renderer prints raw <sub> literally.
    footer = (
        f"\n\n*{data['model']} · {usage['input_tokens']:,}→{usage['output_tokens']:,} tok · "
        f"${usage['cost_usd']:.4f} · {data['retrieval_ms']}ms + {data['generation_ms']:,}ms*"
    )
    await cl.Message(content=data["answer"] + _format_sources(sources) + footer).send()
