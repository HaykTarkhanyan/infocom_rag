"""Draft HARD eval questions by generating against a distractor.

`generate_candidates.py` shows the model one chunk and asks for a question about
it. That produces questions the retriever finds at rank 1 almost every time --
measured 2026-08-02, recall@10 hit 100% for BOTH dense and BM25, so the set had
stopped measuring retrieval at all.

The cause was not corpus size. The corpus contains genuinely confusable
material: 21 of 94 articles have a neighbour above 0.85 cosine, and the closest
pair sits at 0.948 ("ads on opposition-supporting pages" vs "ads on
pro-government pages"). Questions drafted from one randomly chosen chunk simply
never had to tell those apart -- median margin between the right article and the
best wrong one was +0.123, comfortable.

So this script picks the most confusable PAIRS and shows the model both, asking
for a question answerable from A and NOT from B. The distractor is what forces a
discriminating question. Difficulty already latent in the corpus, made visible.

Selection is retriever-neutral -- pairs are chosen by article-centroid cosine,
which is a property of the corpus, not of any retriever we are trying to compare.
Filtering on "BM25 gets this wrong" would guarantee dense wins and measure
nothing.

Usage:
    python eval/generate_hard_candidates.py --pairs 8
    python eval/generate_hard_candidates.py --report
"""

import argparse
import asyncio
import json
import logging
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import llm
import retrieval

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("logs/generate_hard_candidates.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

OUT_PATH = Path("eval/hard_candidates.jsonl")
MIN_TOKENS = 200

PROMPT = """You are building a DELIBERATELY HARD evaluation set for a retrieval system over Armenian news analysis from Infocom.

Below are excerpts from TWO different articles that cover very similar subject matter. They are easy to confuse.

ARTICLE A
---
%s
---

ARTICLE B (the distractor -- do NOT write questions about this one)
---
%s
---

Write TWO questions in Armenian that ARTICLE A answers and ARTICLE B does NOT.

Requirements, in order of importance:
1. DISCRIMINATING. Someone holding only article B must be unable to answer. The question has to turn on what makes A different from B, not on what they share.
2. SELF-CONTAINED. The reader has not seen either article and has no conversation history. Never write a bare "this", "that", "these", "he" or "they" whose referent is only in the article -- name the person, place, organisation or decision. Use the fewest identifying words that do the job.
3. PARAPHRASED. Ask the way a real reader would, in everyday Armenian. Do NOT reuse article A's distinctive noun phrases, and do not copy its sentence structure. If your question shares long word sequences with the article, rewrite it.
4. One question should turn on a specific detail (a figure, a date, a name). The other should turn on a cause, a consequence, or a position someone took.

For each question give `must_contain`: 1-3 SHORT strings that must appear in any correct answer. Armenian is agglutinative, so use word STEMS, numbers, or proper nouns that will not take case endings -- never a full inflected word form.

Return ONLY valid JSON, no markdown fence:
{"questions": [{"text": "...", "kind": "specific", "must_contain": ["..."], "why_hard": "one sentence: what stops article B from answering this"}, {"text": "...", "kind": "broad", "must_contain": ["..."], "why_hard": "..."}]}"""

BAN_CLAUSE = """

BANNED WORDS. You may NOT use any of these words, or any inflected form of them, in your questions. They are article A's most distinctive vocabulary, and reusing them makes the question findable by simple word matching instead of by understanding:

%s

Refer to those ideas some other way -- a synonym, a description, a broader term. If you cannot ask the question without a banned word, ask a different question."""

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def distinctive_terms(text: str, doc_freq: dict[str, int], n_docs: int,
                      top: int = 18) -> list[str]:
    """Article terms that are rare across the corpus -- its lexical fingerprint.

    Banning these is what forces a real paraphrase. Asking the model nicely to
    "avoid distinctive noun phrases" did not work: measured median overlap 0.56,
    HIGHER than the set it was meant to improve on.
    """
    counts: dict[str, int] = {}
    for tok in TOKEN_RE.findall(text.lower()):
        if len(tok) > 3 and not tok.isdigit():
            counts[tok] = counts.get(tok, 0) + 1
    scored = [(c * math.log(n_docs / (1 + doc_freq.get(t, 0))), t) for t, c in counts.items()]
    scored.sort(reverse=True)
    return [t for _, t in scored[:top]]


def confusable_pairs(n: int) -> list[tuple[dict, dict, float]]:
    """The n most similar article pairs, by centroid cosine.

    One representative chunk per article -- the longest, since a short chunk may
    not contain a claim worth asking about.
    """
    matrix, _ = retrieval._dense_index()
    row_chunks = retrieval._row_to_chunk()

    rows_by_post: dict[int, list[int]] = {}
    for row, chunk in enumerate(row_chunks):
        rows_by_post.setdefault(chunk["post_id"], []).append(row)

    pids = sorted(rows_by_post)
    centroids = np.stack([matrix[rows_by_post[p]].mean(0) for p in pids])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    sim = centroids @ centroids.T
    np.fill_diagonal(sim, -1.0)

    best_chunk = {}
    for pid, rows in rows_by_post.items():
        usable = [row_chunks[r] for r in rows if row_chunks[r].get("n_tokens", 0) >= MIN_TOKENS]
        pool = usable or [row_chunks[r] for r in rows]
        best_chunk[pid] = max(pool, key=lambda c: c.get("n_tokens", 0))

    # Each ARTICLE may appear in at most one pair. Without this the top pairs
    # collapse into one topic -- the transit/ticketing cluster alone supplied 3
    # of the top 8, and 5 of 7 surviving questions were about the metro. Taking
    # the best pair per article trades a little similarity for topical spread,
    # which is what "a diverse eval set" actually means.
    order = np.dstack(np.unravel_index(np.argsort(-sim, axis=None), sim.shape))[0]
    out, used = [], set()
    for i, j in order:
        i, j = int(i), int(j)
        if i in used or j in used:
            continue
        used.update((i, j))
        out.append((best_chunk[pids[i]], best_chunk[pids[j]], float(sim[i, j])))
        if len(out) >= n:
            break
    return out


async def draft(a: dict, b: dict, similarity: float,
                banned: list[str] | None = None) -> list[dict]:
    prompt = PROMPT % (a["text"], b["text"])
    if banned:
        prompt += BAN_CLAUSE % ", ".join(banned)
    response = await llm.call(
        [{"role": "user", "content": prompt}],
        role="eval_generate_hard",
    )
    raw = re.sub(r"^```(?:json)?|```$", "", response.content.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("pair %s/%s: non-JSON response, skipping", a["post_id"], b["post_id"])
        return []

    out = []
    for q in parsed.get("questions", []):
        if not q.get("text"):
            continue
        out.append({
            "text": q["text"].strip(),
            "kind": q.get("kind", "specific"),
            "must_contain": [s for s in q.get("must_contain", []) if s],
            "why_hard": q.get("why_hard", ""),
            "source_post_id": a["post_id"],
            "source_chunk_id": a["chunk_id"],
            "source_title": a["title"],
            "source_url": a["url"],
            "published": a["published"],
            "distractor_post_id": b["post_id"],
            "distractor_title": b["title"],
            "pair_similarity": round(similarity, 4),
            "cost_usd": response.usage.cost_usd / 2,
        })
    return out


async def generate(n_pairs: int, ban: bool) -> None:
    pairs = confusable_pairs(n_pairs)
    logger.info("Drafting from the %d most confusable article pairs "
                "(similarity %.3f down to %.3f)",
                len(pairs), pairs[0][2], pairs[-1][2])
    for a, b, s in pairs:
        logger.info("  %.3f  %s  ||  %s", s, a["title"][:44], b["title"][:34])

    banned_by_pair: list[list[str] | None] = [None] * len(pairs)
    if ban:
        all_chunks = retrieval._chunks()
        doc_freq: dict[str, int] = {}
        for chunk in all_chunks:
            for tok in set(TOKEN_RE.findall(chunk["text"].lower())):
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
        banned_by_pair = [distinctive_terms(a["text"], doc_freq, len(all_chunks))
                          for a, _, _ in pairs]
        logger.info("Banning each article's distinctive terms, e.g. %s",
                    ", ".join(banned_by_pair[0][:6]))

    results = await asyncio.gather(
        *(draft(a, b, s, banned) for (a, b, s), banned in zip(pairs, banned_by_pair)),
        return_exceptions=True)
    candidates: list[dict] = []
    for (a, b, _), result in zip(pairs, results):
        if isinstance(result, BaseException):
            logger.error("pair %s/%s failed: %s", a["post_id"], b["post_id"], result)
            continue
        candidates.extend(result)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    total = sum(c["cost_usd"] for c in candidates)
    logger.info("Wrote %d candidates from %d pairs to %s ($%.4f)",
                len(candidates), len(pairs), OUT_PATH, total)
    logger.info("DRAFTS. Score them with --report, then curate the survivors by hand.")


def report() -> None:
    """Score each candidate: lexical overlap, and where the retrievers rank it.

    Both retrievers are reported side by side ON PURPOSE. A candidate that dense
    finds and BM25 misses is the interesting kind; one that both find at rank 1
    is another easy question and should be dropped.
    """
    if not OUT_PATH.exists():
        print(f"{OUT_PATH} not found -- run without --report first")
        sys.exit(1)
    candidates = [json.loads(line) for line in OUT_PATH.open(encoding="utf-8") if line.strip()]
    chunks = {c["chunk_id"]: c for c in retrieval._chunks()}

    print(f"{'overlap':>7} {'dense':>6} {'bm25':>6} {'margin':>7}  question")
    print("-" * 96)
    overlaps, dense_r1, bm25_r1 = [], 0, 0
    for cand in candidates:
        source = chunks.get(cand["source_chunk_id"])
        qt = tokens(cand["text"])
        ov = len(qt & tokens(source["text"])) / max(1, len(qt))
        overlaps.append(ov)

        expected = {cand["source_post_id"]}
        d_hits = retrieval.search_dense(cand["text"], 10, max_distance=2.0)
        b_hits = retrieval.search_bm25(cand["text"], 10)

        def rank(hits, wanted):
            """Rank of the first hit from a wanted article, deduped by article.

            `wanted` is a parameter rather than a closure over the loop variable:
            the closure form works only because it is called in the same
            iteration, and silently breaks the moment it is deferred.
            """
            seen = []
            for h in hits:
                if h.post_id not in seen:
                    seen.append(h.post_id)
            return next((i for i, p in enumerate(seen, 1) if p in wanted), None)

        dr, br = rank(d_hits, expected), rank(b_hits, expected)
        dense_r1 += dr == 1
        bm25_r1 += br == 1

        # margin: how far the right article beats the DISTRACTOR specifically
        best = {}
        for h in d_hits:
            best[h.post_id] = max(best.get(h.post_id, -9), h.score)
        margin = best.get(cand["source_post_id"], 0) - best.get(cand["distractor_post_id"], 0)

        print(f"{ov:7.2f} {dr!s:>6} {br!s:>6} {margin:+7.3f}  {cand['text'][:56]}")

    n = len(candidates)
    overlaps.sort()
    print(f"\nn={n}   median overlap {overlaps[n // 2]:.2f}")
    print(f"found at rank 1:  dense {dense_r1}/{n}   bm25 {bm25_r1}/{n}")
    print("\nDrop candidates both retrievers rank 1 with high overlap -- those are")
    print("just more easy questions. Keep low overlap and/or a small margin.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft hard eval questions using distractors")
    parser.add_argument("--pairs", type=int, default=8,
                        help="How many confusable article pairs to draft from")
    parser.add_argument("--report", action="store_true",
                        help="Score existing candidates instead of generating")
    parser.add_argument("--ban-terms", action="store_true",
                        help="Forbid each article's distinctive vocabulary, forcing "
                             "a real paraphrase (asking politely did not work)")
    args = parser.parse_args()

    if args.report:
        report()
    else:
        asyncio.run(generate(args.pairs, args.ban_terms))


if __name__ == "__main__":
    main()
