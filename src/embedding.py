"""ATE-2 embeddings: mean-pool, L2-normalise, e5 prefixes.

The three things that must be right, and were wrong or missing in the archived
prototype:

1. **Prefixes.** e5-family models are trained with `query: ` and `passage: `
   prefixes and degrade measurably without them. ATE-2 inherits the convention.
   Both come from config.toml so the query and index sides cannot drift apart.
2. **Mean pooling over the attention mask**, not over padding. Confirmed by
   ATE-2-large's own `1_Pooling/config.json`: `pooling_mode_mean_tokens`.
3. **Internal batching.** The caller passes a whole corpus; this splits it into
   fixed batches. The prototype ran every text through in one forward pass, which
   is an OOM waiting to happen on a 16 GB laptop.

Truncation is *not* silent here. Text longer than the model's window means the
chunker failed, so it raises rather than quietly dropping the tail -- the exact
bug that made the prototype useless.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config import settings

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 16


@dataclass
class EmbedStats:
    texts: int
    batches: int
    seconds: float

    @property
    def per_second(self) -> float:
        return self.texts / self.seconds if self.seconds else 0.0


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average token vectors, ignoring padding."""
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class Embedder:
    """Loads ATE-2 once and embeds text. Not thread-safe; one per process."""

    def __init__(self, model_name: str | None = None, device: str | None = None,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        self.model_name = model_name or settings.embedding.model
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading %s on %s", self.model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)
        self.max_tokens = min(settings.embedding.max_tokens,
                              int(self.tokenizer.model_max_length))
        logger.info("Loaded: %d-dim, max %d tokens", self.dim, self.max_tokens)

    def _encode(self, texts: list[str]) -> torch.Tensor:
        batch = self.tokenizer(
            texts, padding=True, truncation=False, return_tensors="pt",
        )
        length = batch["input_ids"].shape[1]
        if length > self.max_tokens:
            raise ValueError(
                f"A text tokenizes to {length} tokens, over the {self.max_tokens} "
                "limit. Chunking should have prevented this -- refusing to "
                "truncate silently."
            )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            out = self.model(**batch)
        pooled = _mean_pool(out.last_hidden_state, batch["attention_mask"])
        return F.normalize(pooled, p=2, dim=1)

    def embed(self, texts: list[str], prefix: str) -> tuple[torch.Tensor, EmbedStats]:
        import time
        if not texts:
            return torch.empty(0, self.dim), EmbedStats(0, 0, 0.0)

        prefixed = [f"{prefix}{t}" for t in texts]
        chunks: list[torch.Tensor] = []
        start = time.perf_counter()
        for i in range(0, len(prefixed), self.batch_size):
            chunks.append(self._encode(prefixed[i:i + self.batch_size]))
        elapsed = time.perf_counter() - start

        stats = EmbedStats(len(texts), len(chunks), elapsed)
        logger.info("Embedded %d texts in %d batches, %.1fs (%.1f/s)",
                    stats.texts, stats.batches, stats.seconds, stats.per_second)
        return torch.cat(chunks), stats

    def embed_passages(self, texts: list[str]) -> tuple[torch.Tensor, EmbedStats]:
        return self.embed(texts, settings.embedding.passage_prefix)

    def embed_query(self, text: str) -> torch.Tensor:
        vectors, _ = self.embed([text], settings.embedding.query_prefix)
        return vectors[0]
