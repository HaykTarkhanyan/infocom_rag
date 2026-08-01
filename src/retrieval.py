"""Keyword retrieval over the chunk corpus.

**This is a stopgap.** The real retriever is dense vectors from ATE-2, which is
blocked on downloading weights. BM25 works today with no model, no vector store
and no GPU, and it is genuinely useful for Armenian proper nouns and institution
names -- the eventual design is hybrid (BM25 + dense) anyway, so this is the
first half of that rather than throwaway work.

Known weakness: Armenian is agglutinative and there is no stemmer here, so
`ժողովը` (with the definite article) and `ժողով` are different terms. Dense
retrieval is what fixes that; do not paper over it with hand-written suffix
rules.

The public surface is `search(query, top_k) -> list[Hit]`. Everything downstream
depends only on that, so swapping in dense or hybrid retrieval later touches this
file alone.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

CHUNKS_PATH = Path("data/chunks.jsonl")

# Split on anything that is not a letter or digit. \w with re.UNICODE keeps
# Armenian letters, which is the whole point -- an ASCII-only pattern would
# silently tokenize Armenian into nothing.
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class Hit:
    """One retrieved chunk. `score` is BM25; higher is better (unlike distance)."""
    chunk_id: str
    post_id: int
    url: str
    title: str
    heading: str | None
    text: str
    published: str
    authors: list[str] = field(default_factory=list)
    infotags: list[str] = field(default_factory=list)
    n_tokens: int = 0
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "post_id": self.post_id,
            "url": self.url,
            "title": self.title,
            "heading": self.heading,
            "published": self.published,
            "authors": self.authors,
            "infotags": self.infotags,
            "n_tokens": self.n_tokens,
            "score": round(self.score, 4),
            "text": self.text,
        }


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def _index() -> tuple[BM25Okapi, list[dict]]:
    """Build the BM25 index once per process. 969 chunks is trivial in memory."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. Run src/fetch_articles.py then src/chunking.py."
        )
    chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8") if line.strip()]
    if not chunks:
        raise ValueError(f"{CHUNKS_PATH} is empty")

    corpus = [tokenize(c["text"]) for c in chunks]
    logger.info("BM25 index built over %d chunks", len(chunks))
    return BM25Okapi(corpus), chunks


def search(query: str, top_k: int = 10, min_score: float = 0.0) -> list[Hit]:
    """Return the top_k chunks for *query*, best first.

    `min_score` drops weak matches. BM25 scores are unbounded and corpus-relative,
    so there is no principled default -- 0.0 keeps everything and lets the caller
    (or the eval) decide.
    """
    if not query.strip():
        return []

    bm25, chunks = _index()
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    hits = []
    for i in ranked:
        if scores[i] <= min_score:
            continue
        c = chunks[i]
        hits.append(Hit(
            chunk_id=c["chunk_id"], post_id=c["post_id"], url=c["url"],
            title=c["title"], heading=c.get("heading"), text=c["text"],
            published=c["published"], authors=c.get("authors", []),
            infotags=c.get("infotags", []), n_tokens=c.get("n_tokens", 0),
            score=float(scores[i]),
        ))
    return hits


def format_context(hits: list[Hit]) -> str:
    """Render hits as the numbered excerpts the system prompt expects.

    The numbering here is what the model cites as [1], [2], so it must match the
    order the caller shows the user.
    """
    blocks = []
    for i, hit in enumerate(hits, 1):
        header = f"[{i}] {hit.title}"
        if hit.heading and hit.heading != hit.title:
            header += f" — {hit.heading}"
        header += f" ({hit.published[:10]}, {hit.url})"
        blocks.append(f"{header}\n{hit.text}")
    return "\n\n".join(blocks)


def corpus_stats() -> dict:
    _, chunks = _index()
    posts = {c["post_id"] for c in chunks}
    return {
        "chunks": len(chunks),
        "articles": len(posts),
        "tokens": sum(c.get("n_tokens", 0) for c in chunks),
    }
