"""Verify the GPU-built index matches what local queries will produce.

`src/embed_corpus_colab.py` is self-contained -- it runs on a bare Colab VM and
cannot import `src/embedding.py`, so the pooling math exists in two places. If
those drift, the index and the queries live in different spaces and retrieval
degrades in a way that looks like "the model is bad" rather than "the code is
inconsistent".

This re-embeds a sample of chunks locally and compares against the stored
vectors. Cosine should be ~1.0. Small deviations (~1e-3) are expected from
GPU/CPU float differences and are harmless for ranking; anything larger means the
implementations actually diverge.

Usage:
    python research/verify_embedding_parity.py                     # base
    python research/verify_embedding_parity.py --model large -n 8
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedding import Embedder  # noqa: E402

CHUNKS_PATH = Path("data/chunks.jsonl")
REPOS = {
    "base": "Metric-AI/armenian-text-embeddings-2-base",
    "large": "Metric-AI/armenian-text-embeddings-2-large",
}
# Must match src/embed_corpus_colab.py.
PASSAGE_PREFIX = "passage: "


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local/GPU embedding parity")
    parser.add_argument("--model", choices=list(REPOS), default="base")
    parser.add_argument("--vectors", help="Path to vectors_<model>.npz")
    parser.add_argument("-n", type=int, default=5, help="Chunks to re-embed locally")
    parser.add_argument("--tolerance", type=float, default=0.999,
                        help="Minimum acceptable cosine (default 0.999)")
    args = parser.parse_args()

    vectors_path = Path(args.vectors or f"data/vectors_{args.model}.npz")
    if not vectors_path.exists():
        print(f"{vectors_path} not found -- run the Colab job and download it first.")
        sys.exit(1)

    stored = np.load(vectors_path, allow_pickle=True)
    matrix = stored["vectors"]
    chunk_ids = list(stored["chunk_ids"])
    print(f"index    : {vectors_path}")
    print(f"           {matrix.shape[0]} vectors x {matrix.shape[1]} dims")
    print(f"           model={stored['model']} prefix={stored['passage_prefix']!r}")

    if str(stored["model"]) != REPOS[args.model]:
        print(f"WARNING: index was built with {stored['model']}, "
              f"comparing against {REPOS[args.model]}")
    if str(stored["passage_prefix"]) != PASSAGE_PREFIX:
        print(f"WARNING: index prefix {stored['passage_prefix']!r} != {PASSAGE_PREFIX!r} "
              "-- queries and passages are in different spaces")

    chunks = {c["chunk_id"]: c for c in
              (json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8") if line.strip())}

    # Spread the sample across the corpus rather than taking the first N, so a
    # drift that only affects long or heading-bearing chunks still shows up.
    step = max(1, len(chunk_ids) // args.n)
    sample_ids = chunk_ids[::step][:args.n]
    texts = [chunks[cid]["text"] for cid in sample_ids]

    embedder = Embedder(REPOS[args.model])
    local, _ = embedder.embed(texts, PASSAGE_PREFIX)
    local_np = local.cpu().numpy()

    print()
    print(f"{'chunk_id':<22} {'cosine':>8}  {'tokens':>6}")
    print("-" * 40)
    worst = 1.0
    for i, cid in enumerate(sample_ids):
        gpu_vec = matrix[chunk_ids.index(cid)]
        cos = float(np.dot(gpu_vec, local_np[i]))
        worst = min(worst, cos)
        flag = "" if cos >= args.tolerance else "   <-- DRIFT"
        print(f"{cid:<22} {cos:8.6f}  {chunks[cid]['n_tokens']:>6}{flag}")

    print()
    print(f"worst cosine: {worst:.6f}  (tolerance {args.tolerance})")
    if worst < args.tolerance:
        print("FAIL: local and GPU embeddings diverge. The pooling math, prefix or "
              "max_tokens differ between src/embedding.py and src/embed_corpus_colab.py.")
        sys.exit(1)
    print("PASS: the index and local queries are in the same space.")


if __name__ == "__main__":
    main()
