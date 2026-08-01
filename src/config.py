"""Load pinned settings from config.toml and secrets from .env.

Nothing in src/ should hardcode a model, temperature, threshold or prompt --
import from here so every knob is versioned in one diffable file.

Reads are strict: a missing key raises at import rather than defaulting, so a
typo in config.toml fails loudly instead of silently changing behaviour.
"""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


def _load() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.toml not found at {CONFIG_PATH}")
    with CONFIG_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _require(table: dict, key: str, where: str):
    """Fetch a key or raise -- no silent defaults for pinned settings."""
    if key not in table:
        raise KeyError(f"config.toml is missing [{where}] {key}")
    return table[key]


@dataclass(frozen=True)
class Generation:
    model: str
    temperature: float
    top_p: float
    top_k: int
    max_output_tokens: int
    fallback: list[str]
    pin_provider: str


@dataclass(frozen=True)
class Retrieval:
    retriever: str
    top_k: int
    max_distance: float


@dataclass(frozen=True)
class Embedding:
    model: str
    max_tokens: int
    overlap_sentences: int
    query_prefix: str
    passage_prefix: str


@dataclass(frozen=True)
class Logging:
    llm_ledger: str


@dataclass(frozen=True)
class Settings:
    generation: Generation
    retrieval: Retrieval
    embedding: Embedding
    logging: Logging
    system_prompt: str


def _build() -> Settings:
    raw = _load()

    gen = _require(raw, "generation", "root")
    ret = _require(raw, "retrieval", "root")
    emb = _require(raw, "embedding", "root")
    log = _require(raw, "logging", "root")
    prompt = _require(raw, "prompt", "root")

    settings = Settings(
        generation=Generation(
            model=_require(gen, "model", "generation"),
            temperature=float(_require(gen, "temperature", "generation")),
            top_p=float(_require(gen, "top_p", "generation")),
            top_k=int(_require(gen, "top_k", "generation")),
            max_output_tokens=int(_require(gen, "max_output_tokens", "generation")),
            fallback=list(_require(gen, "fallback", "generation")),
            pin_provider=_require(gen, "pin_provider", "generation"),
        ),
        retrieval=Retrieval(
            retriever=_require(ret, "retriever", "retrieval"),
            top_k=int(_require(ret, "top_k", "retrieval")),
            max_distance=float(_require(ret, "max_distance", "retrieval")),
        ),
        embedding=Embedding(
            model=_require(emb, "model", "embedding"),
            max_tokens=int(_require(emb, "max_tokens", "embedding")),
            overlap_sentences=int(_require(emb, "overlap_sentences", "embedding")),
            query_prefix=_require(emb, "query_prefix", "embedding"),
            passage_prefix=_require(emb, "passage_prefix", "embedding"),
        ),
        logging=Logging(llm_ledger=_require(log, "llm_ledger", "logging")),
        system_prompt=_require(prompt, "system", "prompt").strip(),
    )

    # Bounds worth catching at load time rather than as a confusing API error.
    if not 0.0 <= settings.generation.temperature <= 2.0:
        raise ValueError(f"temperature must be 0.0-2.0, got {settings.generation.temperature}")
    if settings.retrieval.top_k < 1:
        raise ValueError(f"retrieval.top_k must be >= 1, got {settings.retrieval.top_k}")
    if settings.retrieval.retriever not in ("dense", "bm25"):
        raise ValueError(
            f"retrieval.retriever must be 'dense' or 'bm25', "
            f"got {settings.retrieval.retriever!r}"
        )
    if not 0.0 <= settings.retrieval.max_distance <= 2.0:
        raise ValueError(
            f"retrieval.max_distance must be 0.0-2.0, got {settings.retrieval.max_distance}"
        )
    if settings.embedding.max_tokens > 512:
        raise ValueError(
            f"embedding.max_tokens is {settings.embedding.max_tokens}, but ATE-2 "
            "(XLM-R) caps at 512 -- anything higher would be silently truncated"
        )
    return settings


settings = _build()


def openrouter_key() -> str:
    """Read the API key from the environment. Raises if unset."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to .env "
            "(get one at https://openrouter.ai/keys)."
        )
    return key
