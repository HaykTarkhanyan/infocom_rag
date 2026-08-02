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

# Turns kept in the session. The API shows the rewriter only the last
# `rewrite.max_turns`, so this is a small buffer above that, not a second policy.
HISTORY_LIMIT = 8


APP_PASSWORD = os.environ.get("APP_PASSWORD")

# Registered CONDITIONALLY, and that is load-bearing: `@cl.password_auth_callback`
# takes effect at import, and merely applying it makes Chainlit demand
# CHAINLIT_AUTH_SECRET or refuse to boot --
#   ValueError: You must provide a JWT secret in the environment to use authentication
# An earlier version decorated unconditionally and checked APP_PASSWORD inside the
# function, which crashed the container on startup and would have broken local
# development too. Guarding the decorator itself keeps dev frictionless while the
# deployment stays gated.
if APP_PASSWORD:
    @cl.password_auth_callback
    def auth(username: str, password: str) -> cl.User | None:
        """Gate the UI behind a shared password.

        Deployed publicly without this, anyone who finds the URL spends the
        OpenRouter key -- there is no per-user cap or rate limit yet.

        Uses `secrets.compare_digest` rather than `==` so the comparison does not
        leak the password's length through timing.
        """
        if secrets.compare_digest(password, APP_PASSWORD):
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


@cl.set_starters
async def starters() -> list[cl.Starter]:
    """Clickable example questions on the empty chat screen.

    Every one of these was checked against the live index before being offered:
    all retrieve their correct article at rank 1, at cosine distance 0.21-0.34
    against a 0.55 cutoff. A suggested question that returns nothing would fail
    on a user's very first click, which is the worst possible first impression --
    so if the corpus changes, re-run the check rather than assuming these still
    land.

    Deliberately spread across four different subject areas, so the starters
    advertise the corpus's actual breadth rather than four flavours of one topic.
    """
    return [
        cl.Starter(
            label="Աշտարակի գնումները",
            message="Ինչպե՞ս է Աշտարակի համայնքը գնումներ կատարել ավագանու անդամի ընկերությունից",
        ),
        cl.Starter(
            label="44-օրյա պատերազմի գործերը",
            message="Քանի՞ քրեական գործ է հարուցվել 44-օրյա պատերազմին առնչվող դեպքերով",
        ),
        cl.Starter(
            label="Անկանխիկ կենսաթոշակներ",
            message="Ինչո՞ւ են կենսաթոշակառուները դժվարությամբ կանխիկացնում իրենց թոշակը",
        ),
        cl.Starter(
            label="IMEI գրանցում",
            message="Ի՞նչ է նախատեսում հեռախոսների IMEI գրանցման նոր համակարգը",
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    """Speak ONLY when something is wrong.

    The welcome text lives in chainlit.md, not here, and that is load-bearing:
    Chainlit draws the README and the starters only while the thread is empty, so
    any message sent here silently suppresses BOTH. That is exactly what happened
    on the first deploy -- /project/settings returned all four starters and the
    screen showed none of them.

    Staying silent on the happy path is therefore the feature. An unreachable API
    is worth breaking that rule for: without it the user gets a welcome screen
    and then an error on their first question, with no clue which part is broken.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            (await client.get(f"{API_URL}/health")).raise_for_status()
    except httpx.HTTPError as exc:
        await cl.Message(
            content=(
                f"⚠️ **Չհաջողվեց կապվել API-ի հետ ({API_URL}).**\n\n"
                "Հարցերը հիմա չեն աշխատի։ Գործարկեք API-ն՝\n"
                "```\npython -m uvicorn api:app --app-dir src --port 8000\n```\n\n"
                f"`{exc}`"
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

    # Prior turns, oldest first. Kept per-session in memory rather than read back
    # from Neon: the DB is the durable record, but a chat needs the last few
    # turns synchronously and a query per message would be pure latency.
    history: list[dict] = cl.user_session.get("history") or []

    async with cl.Step(name="Retrieval", type="retrieval") as retrieval_step:
        retrieval_step.input = question
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{API_URL}/ask",
                    json={"question": question, "history": history},
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

        # Show the rewritten query FIRST when it differs. What was searched is
        # not always what was typed, and a silent substitution would be its own
        # version of the bug rewriting exists to fix.
        rewrite_note = ""
        if data.get("rewritten"):
            rewrite_note = (
                f"follow-up resolved for search ({data['rewrite_ms']} ms):\n"
                f"  typed:    {question}\n"
                f"  searched: {data['question_used']}\n\n"
            )

        retrieval_step.output = rewrite_note + ("\n".join(
            f"[{s['n']}] {score_label(s)}  {s['n_tokens']:>3} tok  {s['title'][:52]}"
            + (f"\n      § {s['heading'][:56]}" if s.get("heading") else "")
            for s in sources
        ) or "no hits")
        retrieval_step.metadata = {
            "n_sources": len(sources),
            "retrieval_ms": data["retrieval_ms"],
            "chunk_ids": [s["chunk_id"] for s in sources],
            "retriever": data["config"]["retriever"],
            "embedding_model": data["config"].get("embedding_model"),
            "max_distance": data["config"].get("max_distance"),
            "question_typed": question,
            "question_searched": data.get("question_used", question),
            "rewritten": data.get("rewritten", False),
            "rewrite_ms": data.get("rewrite_ms", 0),
            "history_turns": len(history),
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

    # Record the turn for the next follow-up. Stored under the question the user
    # TYPED, not the rewritten one -- the rewriter is resolving references
    # against a human conversation, and replacing what was said with a machine
    # paraphrase would compound each rewrite on the last.
    #
    # Trimmed to the configured window plus a little slack; the API trims again,
    # so an over-long list here costs memory, never tokens.
    history.append({"question": question, "answer": data["answer"]})
    cl.user_session.set("history", history[-HISTORY_LIMIT:])

    # Plain markdown, not HTML: Chainlit's renderer prints raw <sub> literally.
    rewrite_bit = f" · rewrite {data['rewrite_ms']}ms" if data.get("rewritten") else ""
    footer = (
        f"\n\n*{data['model']} · {usage['input_tokens']:,}→{usage['output_tokens']:,} tok · "
        f"${usage['cost_usd']:.4f} · {data['retrieval_ms']}ms + {data['generation_ms']:,}ms"
        f"{rewrite_bit}*"
    )
    await cl.Message(content=data["answer"] + _format_sources(sources) + footer).send()
