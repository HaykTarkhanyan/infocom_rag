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
    """Warm the BM25 index at startup, and close the shared HTTP client at shutdown.

    Building the index takes a second or two; doing it on the first request would
    make one unlucky user pay for it, and under concurrency several requests
    would race to build it at once.
    """
    await asyncio.to_thread(retrieval.corpus_stats)
    logger.info("Retrieval index warmed")
    yield
    await llm.close_client()
    logger.info("HTTP client closed")


app = FastAPI(
    title="infocom-rag",
    description="RAG over infocom.am Armenian long-form journalism",
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    # Overrides are explicit and echoed back in `config`, so a stored answer
    # still explains which settings produced it.
    top_k: int | None = Field(default=None, ge=1, le=50)
    min_score: float = 0.0


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
        "retriever": "bm25",
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    top_k = req.top_k or settings.retrieval.top_k

    start = time.monotonic_ns()
    # BM25 scoring is CPU-bound and blocking, so it goes to a worker thread
    # rather than stalling the event loop for every other in-flight request.
    hits = await asyncio.to_thread(
        retrieval.search, req.question, top_k, req.min_score
    )
    retrieval_ms = (time.monotonic_ns() - start) // 1_000_000

    if not hits:
        raise HTTPException(
            status_code=404,
            detail="No chunks matched the query. Try different wording.",
        )

    context = retrieval.format_context(hits)
    try:
        response = await llm.answer(req.question, context)
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
            cost_usd=response.usage.cost_usd,
        ),
        model=response.model,
        config={
            "model": settings.generation.model,
            "temperature": settings.generation.temperature,
            "top_k_retrieval": top_k,
            "min_score": req.min_score,
            "retriever": "bm25",
        },
        retrieval_ms=retrieval_ms,
        generation_ms=response.latency_ms,
    )
