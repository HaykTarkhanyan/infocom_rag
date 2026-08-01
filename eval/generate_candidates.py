"""Draft candidate eval questions from known chunks.

Generating a question *from* a chunk means the correct source is known by
construction -- that is where ground truth for retrieval recall comes from,
without anyone hand-labelling 969 chunks.

The obvious hazard: a model asked to write a question about a passage tends to
reuse that passage's vocabulary, which makes lexical retrieval look far better
than it is and makes dense retrieval look easy. Two mitigations:

  1. The prompt demands paraphrase and forbids copying distinctive phrases.
  2. `--report` measures token overlap between each question and its source
     chunk, so a set that is too easy is visible rather than assumed away.

Output is JSONL of CANDIDATES. They are not the eval set -- they need reading and
curating into eval/questions.toml. Anything auto-generated and auto-accepted
measures the generator, not the system.

Usage:
    python eval/generate_candidates.py --n 30
    python eval/generate_candidates.py --report
"""

import argparse
import asyncio
import json
import logging
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import llm

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("logs/generate_candidates.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

CHUNKS_PATH = Path("data/chunks.jsonl")
OUT_PATH = Path("eval/candidates.jsonl")
SEED = 509

# Long enough to contain a real claim; not so long the question could be about
# anything in it.
MIN_TOKENS = 200

PROMPT = """You are helping build an evaluation set for a retrieval system over Armenian news analysis articles from Infocom.

Below is one excerpt from an article. Write TWO questions in Armenian that this excerpt answers.

Requirements:
- Questions must be answerable from THIS excerpt alone.
- Write them the way a reader would actually ask, NOT by copying the excerpt's phrasing. Paraphrase. Avoid reusing the excerpt's distinctive noun phrases verbatim.
- One question should be specific (a figure, a date, a name, a decision). One should be broader (a cause, a consequence, a position someone took).
- Do not ask meta questions about "the article" or "the excerpt". Ask about the subject matter.
- For each question also give `must_contain`: 1-3 short strings that MUST appear in any correct answer (a number, a name, a key term). Keep them short and exact.

Return ONLY valid JSON, no markdown fence:
{"questions": [{"text": "...", "kind": "specific", "must_contain": ["..."]}, {"text": "...", "kind": "broad", "must_contain": ["..."]}]}

Excerpt:
---
%s
---"""

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} not found -- run src/chunking.py first")
    return [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8") if line.strip()]


def pick(chunks: list[dict], n: int) -> list[dict]:
    """One chunk per article, spread across the corpus, biased to dense ones."""
    rng = random.Random(SEED)
    by_post: dict[int, list[dict]] = {}
    for chunk in chunks:
        if chunk.get("n_tokens", 0) >= MIN_TOKENS:
            by_post.setdefault(chunk["post_id"], []).append(chunk)
    if not by_post:
        raise ValueError(f"No chunks with >= {MIN_TOKENS} tokens")

    posts = sorted(by_post)
    rng.shuffle(posts)
    return [rng.choice(by_post[p]) for p in posts[:n]]


async def draft(chunk: dict) -> list[dict]:
    response = await llm.call(
        [{"role": "user", "content": PROMPT % chunk["text"]}],
        role="eval_generate",
    )
    raw = response.content.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Chunk %s: model returned non-JSON, skipping", chunk["chunk_id"])
        return []

    out = []
    for q in parsed.get("questions", []):
        if not q.get("text"):
            continue
        out.append({
            "text": q["text"].strip(),
            "kind": q.get("kind", "specific"),
            "must_contain": [s for s in q.get("must_contain", []) if s],
            "source_post_id": chunk["post_id"],
            "source_chunk_id": chunk["chunk_id"],
            "source_title": chunk["title"],
            "source_url": chunk["url"],
            "published": chunk["published"],
            "cost_usd": response.usage.cost_usd / 2,
        })
    return out


async def generate(n: int) -> None:
    chunks = load_chunks()
    picked = pick(chunks, n)
    logger.info("Drafting from %d chunks (one per article)", len(picked))

    results = await asyncio.gather(*(draft(c) for c in picked), return_exceptions=True)
    candidates: list[dict] = []
    for chunk, result in zip(picked, results):
        if isinstance(result, BaseException):
            logger.error("Chunk %s failed: %s", chunk["chunk_id"], result)
            continue
        candidates.extend(result)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    total = sum(c["cost_usd"] for c in candidates)
    logger.info("Wrote %d candidates from %d articles to %s ($%.4f)",
                len(candidates), len(picked), OUT_PATH, total)
    logger.info("These are DRAFTS -- curate them into eval/questions.toml by hand.")


def report() -> None:
    """Measure question/source lexical overlap: high overlap = a too-easy set."""
    if not OUT_PATH.exists():
        print(f"{OUT_PATH} not found -- run without --report first")
        sys.exit(1)
    candidates = [json.loads(l) for l in OUT_PATH.open(encoding="utf-8") if l.strip()]
    by_chunk = {c["chunk_id"]: c for c in load_chunks()}

    print(f"{'overlap':>8}  {'kind':<9} question")
    print("-" * 78)
    overlaps = []
    for cand in candidates:
        source = by_chunk.get(cand["source_chunk_id"])
        if not source:
            continue
        q_tokens = tokens(cand["text"])
        overlap = len(q_tokens & tokens(source["text"])) / max(1, len(q_tokens))
        overlaps.append(overlap)
        print(f"{overlap:8.2f}  {cand['kind']:<9} {cand['text'][:58]}")

    if overlaps:
        overlaps.sort()
        print()
        print(f"token overlap with source: median {overlaps[len(overlaps)//2]:.2f}  "
              f"max {overlaps[-1]:.2f}")
        print("Above ~0.6 median means the questions largely restate the source and "
              "the set will flatter any retriever. Rewrite the worst offenders.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft eval question candidates")
    parser.add_argument("--n", type=int, default=25, help="Articles to draft from")
    parser.add_argument("--report", action="store_true",
                        help="Report question/source lexical overlap instead")
    args = parser.parse_args()

    if args.report:
        report()
    else:
        asyncio.run(generate(args.n))


if __name__ == "__main__":
    main()
