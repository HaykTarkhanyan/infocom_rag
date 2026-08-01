"""Embed the chunk corpus with both ATE-2 models on a Colab GPU.

Run from WSL (the Colab CLI is Unix-only -- it imports `termios`):

    colab new -s infocom --gpu T4
    colab upload -s infocom data/chunks.jsonl /content/chunks.jsonl
    colab exec -s infocom -f src/embed_corpus_colab.py
    colab download -s infocom /content/vectors_base.npz data/
    colab download -s infocom /content/vectors_large.npz data/
    colab stop -s infocom          # idle VMs burn compute units

Deliberately SELF-CONTAINED: it runs on a bare Colab VM with no access to this
repo, so it cannot import `src/embedding.py`. The pooling math is therefore
duplicated, which is a real risk -- if the two drift, queries embedded locally
stop matching an index embedded here. `verify_parity` in
`research/verify_embedding_parity.py` guards that by embedding identical text on
both sides and asserting cosine ~= 1.

Local CPU measured 1.1 chunks/s (base) and 0.4/s (large); a T4 should be
roughly two orders of magnitude faster.
"""

import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODELS = {
    "base": "Metric-AI/armenian-text-embeddings-2-base",
    "large": "Metric-AI/armenian-text-embeddings-2-large",
}

# Must match config.toml [embedding]. Hardcoded because config.toml is not on
# this VM; any change there has to be mirrored here.
PASSAGE_PREFIX = "passage: "
MAX_TOKENS = 512
BATCH_SIZE = 64          # a T4 handles this comfortably at 512 tokens

CHUNKS_PATH = "/content/chunks.jsonl"


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average token vectors ignoring padding -- ATE-2's pooling_mode_mean_tokens."""
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device == 'cuda' else ''}")

    with open(CHUNKS_PATH, encoding="utf-8") as handle:
        chunks = [json.loads(line) for line in handle if line.strip()]
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    print(f"corpus: {len(texts)} chunks")

    for label, repo in MODELS.items():
        print(f"\n=== {label}: {repo}")
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).to(device).eval()

        vectors = []
        truncated = 0
        start = time.perf_counter()
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = [f"{PASSAGE_PREFIX}{t}" for t in texts[i:i + BATCH_SIZE]]
            batch = tokenizer(batch_texts, padding=True, truncation=True,
                              max_length=MAX_TOKENS, return_tensors="pt")
            # Chunking should guarantee this never fires; count rather than
            # silently accept, so a chunker regression is visible.
            lengths = batch["attention_mask"].sum(dim=1)
            truncated += int((lengths >= MAX_TOKENS).sum())
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.inference_mode():
                out = model(**batch)
            pooled = mean_pool(out.last_hidden_state, batch["attention_mask"])
            vectors.append(F.normalize(pooled, p=2, dim=1).cpu().float().numpy())

            done = min(i + BATCH_SIZE, len(texts))
            if done % (BATCH_SIZE * 4) == 0 or done == len(texts):
                rate = done / (time.perf_counter() - start)
                print(f"  {done}/{len(texts)}  {rate:.1f} chunks/s", flush=True)

        elapsed = time.perf_counter() - start
        matrix = np.vstack(vectors).astype(np.float32)
        out_path = f"/content/vectors_{label}.npz"
        np.savez_compressed(
            out_path,
            vectors=matrix,
            chunk_ids=np.array(chunk_ids, dtype=object),
            model=repo,
            passage_prefix=PASSAGE_PREFIX,
            max_tokens=MAX_TOKENS,
        )
        print(f"  shape {matrix.shape}  {elapsed:.1f}s  ({len(texts)/elapsed:.1f}/s)")
        print(f"  at-limit chunks: {truncated}")
        print(f"  saved {out_path}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
