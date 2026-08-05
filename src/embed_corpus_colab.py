"""Embed the chunk corpus with both ATE-2 models on a Colab GPU.

Run from WSL (the Colab CLI is Unix-only -- it imports `termios`):

    colab new -s infocom --gpu T4
    colab upload -s infocom data/chunks.jsonl /content/chunks.jsonl
    colab exec -s infocom --timeout 5400 -f src/embed_corpus_colab.py
    colab download -s infocom /content/vectors_large.npz data/vectors_large.npz
    colab stop -s infocom          # idle VMs burn compute units

**Run it ATTACHED, with a generous --timeout.** Both failure modes were hit
building the 25,797-chunk news index on 2026-08-04:

  * attached with --timeout 2400 -- the client gave up at exactly 40 minutes and
    took the run with it, at ~23k/25.8k, no artifact written;
  * detached via subprocess to dodge that -- which left the Jupyter KERNEL idle,
    so Colab's idle detection reclaimed the whole VM at ~7k/25.8k and the
    session was lost (404).

Staying attached keeps the kernel visibly busy; the timeout just has to be
bigger than the job. `--timeout` defaults to 30s, so it is never optional here.

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

ALL_MODELS = {
    "base": "Metric-AI/armenian-text-embeddings-2-base",
    "large": "Metric-AI/armenian-text-embeddings-2-large",
}

# Which of the above to build. `colab exec -f` runs this file with no arguments
# on a bare VM, so this is a constant rather than a CLI flag.
#
# large only, by default. config.toml pins large, and the base-vs-large question
# is settled (DECISIONS #6). At 969 chunks building both was free; at 25,797 it
# doubles GPU time and yields a ~100 MB index nothing reads. Set to
# ("base", "large") if a comparison is ever wanted again.
BUILD = ("large",)
MODELS = {k: v for k, v in ALL_MODELS.items() if k in BUILD}

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

        # Batch chunks of SIMILAR LENGTH together. `padding=True` pads to the
        # longest item in each batch, so a 35-token chunk sharing a batch with a
        # 512-token one costs a full 512 either way. Chunk lengths here run
        # 35-512 with a median of 281, so random batching wastes roughly a third
        # of the compute on padding. Purely a scheduling change -- each chunk
        # still gets the identical forward pass -- and the original order is
        # restored below.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        pieces: dict[int, np.ndarray] = {}
        truncated = 0
        start = time.perf_counter()
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            batch_texts = [f"{PASSAGE_PREFIX}{texts[j]}" for j in idx]
            batch = tokenizer(batch_texts, padding=True, truncation=True,
                              max_length=MAX_TOKENS, return_tensors="pt")
            # Chunking should guarantee this never fires; count rather than
            # silently accept, so a chunker regression is visible.
            lengths = batch["attention_mask"].sum(dim=1)
            truncated += int((lengths >= MAX_TOKENS).sum())
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.inference_mode():
                if device == "cuda":
                    # fp16 on the T4's tensor cores. Pooling and normalisation
                    # stay in fp32 below, and the stored vectors are fp32, so the
                    # only fp16 step is the forward pass. Per-element error is
                    # ~1e-3 against retrieval score gaps of ~0.01-0.1 -- but that
                    # is an argument, not evidence, so
                    # research/verify_embedding_parity.py must be re-run after
                    # any index built this way.
                    with torch.autocast("cuda", dtype=torch.float16):
                        out = model(**batch)
                    hidden = out.last_hidden_state.float()
                else:
                    out = model(**batch)
                    hidden = out.last_hidden_state
            pooled = mean_pool(hidden, batch["attention_mask"])
            normed = F.normalize(pooled, p=2, dim=1).cpu().float().numpy()
            for row, j in enumerate(idx):
                pieces[j] = normed[row]

            done = min(i + BATCH_SIZE, len(order))
            if done % (BATCH_SIZE * 4) == 0 or done == len(order):
                rate = done / (time.perf_counter() - start)
                eta = (len(order) - done) / max(rate, 1e-9)
                print(f"  {done}/{len(order)}  {rate:.1f} chunks/s  eta {eta/60:.1f}m",
                      flush=True)

        elapsed = time.perf_counter() - start
        # Restore corpus order so row i still corresponds to chunk_ids[i].
        matrix = np.stack([pieces[i] for i in range(len(texts))]).astype(np.float32)
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
