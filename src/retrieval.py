"""Retrieval over the chunk corpus: dense (ATE-2) or BM25.

`search(query, top_k) -> list[Hit]` is the whole public surface. Everything
downstream depends only on that, which is what let dense retrieval drop in
without touching the API or the UI.

**dense** (default) uses ATE-2-large vectors from `data/vectors_large.npz`, built
by `src/embed_corpus_colab.py`. Chosen on measurement: 5/5 relevant hits in top-5
against BM25's 3/5 and base's 2/5 on a real question.

**bm25** was the pre-embedding stopgap. It is kept because it is the lexical half
of eventual hybrid retrieval, because it needs no model to run, and because
having two retrievers behind one interface is what makes them comparable on the
eval set.

Scores are NOT comparable between the two: BM25 is unbounded and corpus-relative,
dense is cosine in [-1, 1]. `Hit.retriever` says which produced a hit, and
`Hit.distance` is populated for dense only.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from config import settings

logger = logging.getLogger(__name__)

CHUNKS_PATH = Path("data/chunks.jsonl")

# Split on anything that is not a letter or digit. \w with re.UNICODE keeps
# Armenian letters; an ASCII-only pattern would tokenize Armenian into nothing.
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class Hit:
    """One retrieved chunk. `score` is always higher-is-better."""
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
    distance: float | None = None      # dense only: 1 - cosine
    retriever: str = "dense"

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
            "distance": round(self.distance, 4) if self.distance is not None else None,
            "retriever": self.retriever,
            "text": self.text,
        }


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def _chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. Run src/fetch_articles.py then src/chunking.py."
        )
    chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8") if line.strip()]
    if not chunks:
        raise ValueError(f"{CHUNKS_PATH} is empty")
    return chunks


def _vectors_path() -> Path:
    """data/vectors_<base|large>.npz, derived from the configured model."""
    suffix = settings.embedding.model.rsplit("-", 1)[-1]
    return Path(f"data/vectors_{suffix}.npz")


@lru_cache(maxsize=1)
def _dense_index() -> tuple[np.ndarray, dict[str, int]]:
    """Load the vector matrix and a chunk_id -> row map.

    Fails loudly on a model/index mismatch. Silently querying an index built by a
    different model puts the query and the passages in different spaces, which
    degrades results without erroring -- the worst kind of bug here.
    """
    path = _vectors_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with src/embed_corpus_colab.py "
            f"(see its docstring for the Colab commands)."
        )
    data = np.load(path, allow_pickle=True)
    built_with = str(data["model"])
    if built_with != settings.embedding.model:
        raise ValueError(
            f"{path} was built with {built_with}, but config.toml specifies "
            f"{settings.embedding.model}. Re-run the embedding job."
        )

    matrix = data["vectors"].astype(np.float32)
    ids = [str(c) for c in data["chunk_ids"]]
    known = {c["chunk_id"] for c in _chunks()}
    missing = [i for i in ids if i not in known]
    if missing:
        raise ValueError(
            f"{path} references {len(missing)} chunk ids absent from {CHUNKS_PATH} "
            f"(e.g. {missing[:3]}). The corpus was re-chunked after indexing."
        )

    logger.info("Dense index: %d vectors x %d dims (%s)",
                matrix.shape[0], matrix.shape[1], built_with)
    return matrix, {cid: i for i, cid in enumerate(ids)}


@lru_cache(maxsize=1)
def _embedder():
    # Imported lazily: loading torch and a 2 GB model is not something a BM25-only
    # run should pay for.
    from embedding import Embedder
    return Embedder(settings.embedding.model)


@lru_cache(maxsize=1)
def _bm25() -> BM25Okapi:
    corpus = [tokenize(c["text"]) for c in _chunks()]
    logger.info("BM25 index built over %d chunks", len(corpus))
    return BM25Okapi(corpus)


def _hit(chunk: dict, score: float, retriever: str,
         distance: float | None = None) -> Hit:
    return Hit(
        chunk_id=chunk["chunk_id"], post_id=chunk["post_id"], url=chunk["url"],
        title=chunk["title"], heading=chunk.get("heading"), text=chunk["text"],
        published=chunk["published"], authors=chunk.get("authors", []),
        infotags=chunk.get("infotags", []), n_tokens=chunk.get("n_tokens", 0),
        score=score, distance=distance, retriever=retriever,
    )


@lru_cache(maxsize=1)
def _row_to_chunk() -> list[dict]:
    """Row index -> chunk, aligned with the vector matrix.

    Built once. Rebuilding these maps per query was ~2 dict constructions over
    the whole corpus on every request -- unnoticeable at 969 chunks, wasteful at
    the ~40k the full indepth corpus would produce, and pure garbage churn under
    concurrency.
    """
    _, id_to_row = _dense_index()
    by_id = {c["chunk_id"]: c for c in _chunks()}
    ordered: list[dict] = [None] * len(id_to_row)  # type: ignore[list-item]
    for chunk_id, row in id_to_row.items():
        ordered[row] = by_id[chunk_id]
    return ordered


def search_dense(query: str, top_k: int, max_distance: float | None = None) -> list[Hit]:
    """Cosine similarity against the ATE-2 index."""
    max_distance = settings.retrieval.max_distance if max_distance is None else max_distance
    matrix, _ = _dense_index()
    row_chunks = _row_to_chunk()

    query_vec = _embedder().embed_query(query).cpu().numpy()
    sims = matrix @ query_vec           # both sides are L2-normalised

    # argpartition is O(n) and finds the top_k without ordering the rest;
    # argsort would order all ~40k rows to return 10.
    k = min(top_k, sims.shape[0])
    candidates = np.argpartition(-sims, k - 1)[:k]
    ranked = candidates[np.argsort(-sims[candidates])]

    hits: list[Hit] = []
    for row in ranked:
        similarity = float(sims[row])
        distance = 1.0 - similarity
        if distance > max_distance:
            continue
        hits.append(_hit(row_chunks[int(row)], similarity, "dense", distance))
    return hits


def search_bm25(query: str, top_k: int, min_score: float = 0.0) -> list[Hit]:
    """Lexical scoring. `min_score` has no principled default -- BM25 scores are
    unbounded and corpus-relative, so 0.0 keeps everything."""
    chunks = _chunks()
    scores = _bm25().get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [_hit(chunks[i], float(scores[i]), "bm25")
            for i in ranked if scores[i] > min_score]


def search(query: str, top_k: int | None = None, min_score: float = 0.0,
           retriever: str | None = None, max_distance: float | None = None) -> list[Hit]:
    """Retrieve the best chunks for *query*, best first.

    Dispatches on the configured retriever. `min_score` applies to BM25 only and
    `max_distance` to dense only, because the two scales are unrelated.
    """
    if not query.strip():
        return []
    top_k = top_k or settings.retrieval.top_k
    retriever = retriever or settings.retrieval.retriever

    if retriever == "dense":
        return search_dense(query, top_k, max_distance)
    if retriever == "bm25":
        return search_bm25(query, top_k, min_score)
    raise ValueError(f"Unknown retriever {retriever!r}; expected 'dense' or 'bm25'")


def format_context(hits: list[Hit]) -> str:
    """Render hits as the numbered excerpts the system prompt expects.

    The numbering is what the model cites as [1], [2], so it must match the order
    shown to the user.
    """
    blocks = []
    for i, hit in enumerate(hits, 1):
        header = f"[{i}] {hit.title}"
        if hit.heading and hit.heading != hit.title:
            header += f" — {hit.heading}"
        header += f" ({hit.published[:10]}, {hit.url})"
        blocks.append(f"{header}\n{hit.text}")
    return "\n\n".join(blocks)


def warm() -> dict:
    """Load whatever the configured retriever needs. Called from the API lifespan
    so no first request pays for it and concurrent requests cannot race to build it."""
    stats = corpus_stats()
    if settings.retrieval.retriever == "dense":
        _dense_index()
        _embedder()
    else:
        _bm25()
    return stats


def corpus_stats() -> dict:
    chunks = _chunks()
    return {
        "chunks": len(chunks),
        "articles": len({c["post_id"] for c in chunks}),
        "tokens": sum(c.get("n_tokens", 0) for c in chunks),
    }
