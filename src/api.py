"""FastAPI surface for the RAG pipeline.

This is the ONLY entry point. The Chainlit UI and the eval harness both call
`/ask` over HTTP rather than importing the pipeline, so the eval always exercises
the same code path the user does. Scripts that import internals drift from the
real path and then pass while production is broken.

Run:
    python -m uvicorn api:app --app-dir src --reload --port 8000
    curl -s localhost:8000/health
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import llm
import retrieval
from config import settings

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the retriever at startup, close the shared HTTP client at shutdown.

    Doing this on the first request would make one unlucky user wait, and under
    concurrency several requests would race to build the same index.
    """
    # Dense warm-up loads a ~2 GB model, so it must not happen on a request.
    await asyncio.to_thread(retrieval.warm)
    logger.info("Retrieval warmed (%s, %s)",
                settings.retrieval.retriever, settings.embedding.model)
    yield
    await llm.close_client()
    logger.info("HTTP client closed")


app = FastAPI(
    title="infocom-rag",
    description="RAG over infocom.am Armenian long-form journalism",
    version="0.1.0",
    lifespan=lifespan,
)


class Turn(BaseModel):
    question: str
    answer: str = ""


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    # Prior turns, oldest first. OPTIONAL and empty by default, which keeps every
    # existing caller single-turn -- the eval harness deliberately sends none, so
    # its measurements stay comparable with earlier runs.
    history: list[Turn] = Field(default_factory=list)
    # Overrides are explicit and echoed back in `config`, so a stored answer
    # still explains which settings produced it.
    top_k: int | None = Field(default=None, ge=1, le=50)
    min_score: float = 0.0                       # bm25 only
    max_distance: float | None = Field(default=None, ge=0.0, le=2.0)  # dense only
    retriever: str | None = None                 # override the configured default
    rewrite: bool | None = None                  # override [rewrite] enabled


class Source(BaseModel):
    n: int
    chunk_id: str
    post_id: int
    url: str
    title: str
    heading: str | None
    published: str
    authors: list[str]
    infotags: list[str]
    score: float
    distance: float | None
    retriever: str
    n_tokens: int
    text: str


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    cost_usd: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    prompt: str
    usage: Usage
    model: str
    config: dict
    retrieval_ms: int
    generation_ms: int
    # What was ACTUALLY searched and answered. Differs from the user's wording
    # whenever a follow-up was rewritten, and surfacing it is the point: the
    # failure this prevents is invisible, so the substitution must not be.
    question_used: str = ""
    rewritten: bool = False
    rewrite_ms: int = 0


@app.get("/health")
async def health() -> dict:
    """Liveness plus enough context to tell which corpus and model are loaded."""
    try:
        stats = retrieval.corpus_stats()  # cached after lifespan warm-up
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"corpus unavailable: {exc}") from exc
    return {
        "status": "ok",
        "corpus": stats,
        "model": settings.generation.model,
        "retriever": settings.retrieval.retriever,
        "embedding_model": settings.embedding.model,
        "max_distance": settings.retrieval.max_distance,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    top_k = req.top_k or settings.retrieval.top_k

    # --- resolve the question against the conversation, BEFORE retrieval ---
    # A follow-up like "and how many of THOSE went to court?" has no referent on
    # its own. Retrieved as-is it lands on an unrelated article and produces a
    # fluent, correctly-cited answer to a question nobody asked. Rewriting has to
    # happen here rather than at generation time, because retrieval is where the
    # wrong document gets chosen.
    question = req.question
    rewritten = False
    rewrite_ms = 0
    rewrite_cost = 0.0
    use_rewrite = settings.rewrite.enabled if req.rewrite is None else req.rewrite

    if use_rewrite and req.history:
        start = time.monotonic_ns()
        try:
            resolved = await llm.rewrite_question(
                req.question, [t.model_dump() for t in req.history]
            )
        except llm.LLMError as exc:
            # Deliberately fatal. Continuing would retrieve on unresolved
            # pronouns -- the exact silent failure this guards against.
            logger.error("Question rewrite failed: %s", exc)
            raise HTTPException(
                status_code=502, detail=f"Could not resolve the follow-up question: {exc}"
            ) from exc
        rewrite_ms = (time.monotonic_ns() - start) // 1_000_000
        rewrite_cost = resolved.usage.cost_usd
        candidate = resolved.content.strip().strip('"')
        if candidate and candidate != req.question:
            question = candidate
            rewritten = True
            logger.info("Rewrote %r -> %r", req.question, question)

    start = time.monotonic_ns()
    # Retrieval is CPU-bound and blocking either way -- BM25 scoring, or a torch
    # forward pass to embed the query -- so it goes to a worker thread rather
    # than stalling the event loop for every other in-flight request.
    hits = await asyncio.to_thread(
        retrieval.search, question, top_k, req.min_score,
        req.retriever, req.max_distance,
    )
    retrieval_ms = (time.monotonic_ns() - start) // 1_000_000

    if not hits:
        raise HTTPException(
            status_code=404,
            detail=("No chunks passed the retrieval threshold "
                    f"(max_distance={req.max_distance if req.max_distance is not None else settings.retrieval.max_distance}). "
                    "Try different wording, or loosen the threshold."),
        )

    context = retrieval.format_context(hits)
    try:
        response = await llm.answer(question, context)
    except llm.LLMError as exc:
        logger.error("Generation failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

    sources = [
        Source(n=i, **{k: v for k, v in hit.to_dict().items()})
        for i, hit in enumerate(hits, 1)
    ]

    return AskResponse(
        answer=response.content,
        sources=sources,
        prompt=context,
        usage=Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            reasoning_tokens=response.usage.reasoning_tokens,
            cached_tokens=response.usage.cached_tokens,
            # Includes the rewrite call, so the reported cost is what the turn
            # actually cost rather than just its last step.
            cost_usd=response.usage.cost_usd + rewrite_cost,
        ),
        model=response.model,
        config={
            "model": settings.generation.model,
            "temperature": settings.generation.temperature,
            "top_k_retrieval": top_k,
            "retriever": req.retriever or settings.retrieval.retriever,
            "embedding_model": settings.embedding.model,
            "max_distance": (req.max_distance if req.max_distance is not None
                             else settings.retrieval.max_distance),
            "rewrite": use_rewrite,
            "history_turns": len(req.history),
        },
        retrieval_ms=retrieval_ms,
        generation_ms=response.latency_ms,
        question_used=question,
        rewritten=rewritten,
        rewrite_ms=rewrite_ms,
    )
