"""A local tiktokenizer for ATE-2 -- see exactly how Armenian text is tokenized.

Why local rather than a web playground: tiktokenizer.vercel.app cannot load
arbitrary Hugging Face models at all, and Xenova's browser playground loads
ATE-2's real tokenizer.json but tokenizes it differently in JS -- 23 tokens
where the Python tokenizer gives 15 on the same sentence, a 53% overcount.
Chunk budgets computed from that would be wrong, so measure here.

Usage:
    python research/inspect_tokenizer.py "Հայաստանի Ազգային ժողովը"
    python research/inspect_tokenizer.py --chunk 3          # a chunk from data/
    python research/inspect_tokenizer.py --file some.txt
    python research/inspect_tokenizer.py --compare "text"   # ATE-2 vs other models
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from transformers import AutoTokenizer

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "Metric-AI/armenian-text-embeddings-2-base")
CHUNKS_PATH = Path("data/chunks.jsonl")

# SentencePiece marks a word-initial position with U+2581; show it as a space.
SP_SPACE = "▁"


def show(text: str, model: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model)
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)

    print(f"model      : {model}")
    print(f"tokenizer  : {tokenizer.__class__.__name__}  (vocab {tokenizer.vocab_size:,})")
    print(f"characters : {len(text):,}")
    print(f"words      : {len(text.split()):,}")
    print(f"tokens     : {len(ids):,}")
    if text.split():
        print(f"tokens/word: {len(ids) / len(text.split()):.2f}")
    print(f"of 512     : {100 * len(ids) / 512:.0f}%")
    print()

    print("tokens (| = boundary, _ = word start):")
    rendered = "|".join(t.replace(SP_SPACE, "_") for t in tokens)
    print(f"  {rendered}")
    print()
    print("ids:")
    print(f"  {ids}")


def compare(text: str) -> None:
    """Tokenize the same text with several models to show why the choice matters."""
    models = [
        DEFAULT_MODEL,
        "Metric-AI/armenian-text-embeddings-1",
        "intfloat/multilingual-e5-base",
        "bert-base-multilingual-cased",
    ]
    print(f"text: {text[:70]}{'...' if len(text) > 70 else ''}")
    print(f"({len(text)} chars, {len(text.split())} words)")
    print()
    print(f"{'model':46s} {'tokens':>7s} {'tok/word':>9s}")
    print("-" * 66)
    for model in models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model)
            n = len(tokenizer(text, add_special_tokens=True)["input_ids"])
            per_word = n / max(1, len(text.split()))
            print(f"{model:46s} {n:7d} {per_word:9.2f}")
        except Exception as exc:  # noqa: BLE001 - report and continue, this is a diagnostic
            print(f"{model:46s}   FAILED  {type(exc).__name__}: {str(exc)[:40]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ATE-2 tokenization locally")
    parser.add_argument("text", nargs="?", help="Text to tokenize")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--file", help="Read text from a file instead")
    parser.add_argument("--chunk", type=int, help="Index into data/chunks.jsonl")
    parser.add_argument("--compare", action="store_true",
                        help="Compare token counts across models")
    args = parser.parse_args()

    if args.chunk is not None:
        if not CHUNKS_PATH.exists():
            print(f"{CHUNKS_PATH} not found -- run src/chunking.py first", file=sys.stderr)
            sys.exit(1)
        chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8") if line.strip()]
        if not 0 <= args.chunk < len(chunks):
            print(f"--chunk must be 0..{len(chunks) - 1}", file=sys.stderr)
            sys.exit(1)
        chunk = chunks[args.chunk]
        print(f"chunk {chunk['chunk_id']}  ({chunk['url']})")
        print(f"recorded n_tokens: {chunk['n_tokens']}")
        print()
        text = chunk["text"]
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        text = "Հայաստանի Ազգային ժողովը քննարկեց գնումների մասին օրենքի փոփոխությունները։"
        print("(no text given, using a sample)")
        print()

    if args.compare:
        compare(text)
    else:
        show(text, args.model)


if __name__ == "__main__":
    main()
