"""Run the eval set against the live API and report.

Two things are measured SEPARATELY, because they fail differently and the fixes
are unrelated:

  RETRIEVAL  did the right article reach top-k at all?  (recall@k, MRR)
             fixed by: retriever, chunking, thresholds
  GROUNDING  given what was retrieved, is the answer faithful?  (LLM judge)
             fixed by: prompt, model

A system can score well on one and badly on the other, and averaging them into a
single number hides which.

Hits the running API over HTTP rather than importing the pipeline, so the eval
exercises the same path a user does.

Usage:
    python eval/run_eval.py                      # full set
    python eval/run_eval.py --smoke              # priority 5 only
    python eval/run_eval.py --limit 5 --no-judge # cheap retrieval-only pass
    python eval/run_eval.py --retriever bm25     # compare retrievers
"""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import statistics
import sys
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import llm  # noqa: E402

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("logs/run_eval.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

QUESTIONS = Path("eval/questions.toml")
RESULTS = Path("eval/results.jsonl")
JUDGE_CACHE = Path("eval/judge_cache.jsonl")
API_URL = "http://localhost:8000"

# The judge MUST be stronger at the task than the model being graded, per
# _knowledge/02. We grade gpt-5.4-mini, which scores 0.965 on ArmBench reading
# comprehension. gemini-3.1-pro would be 11x cheaper but scores 0.839 -- WORSE at
# Armenian reading than the model it is judging, which defeats the point.
# gpt-5.2-pro (0.973) is the only clearly stronger option.
#
# It costs ~$3.28 for a full 35-question run, which is why judgements are cached
# by (question, answer, judge_model): re-running after a change that does not
# alter answers is then nearly free.
JUDGE_MODEL = "openai/gpt-5.2-pro"


def judge_key(question: dict, answer: str, model: str) -> str:
    payload = json.dumps([question["id"], question.get("grading_notes", ""),
                          answer, model], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def load_judge_cache() -> dict[str, dict]:
    if not JUDGE_CACHE.exists():
        return {}
    cache = {}
    for line in JUDGE_CACHE.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            cache[row["key"]] = row["verdict"]
    return cache


def save_judgement(key: str, verdict: dict) -> None:
    JUDGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with JUDGE_CACHE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": key, "verdict": verdict},
                                ensure_ascii=False) + "\n")

JUDGE_PROMPT = """You are grading a retrieval-augmented answer. Be strict.

QUESTION:
%s

EXCERPTS THE SYSTEM WAS GIVEN:
%s

THE SYSTEM'S ANSWER:
%s

GRADING NOTES:
%s

Grade on two things:

1. grounded: is every factual claim in the answer supported by the excerpts?
   - false if it states facts not present in the excerpts, even if true in reality.
   - false if it attributes a claim to the wrong person or source.

2. correct: does it satisfy the grading notes and actually answer the question?
   - For an unanswerable question, "correct" means the answer DECLINES and says
     the excerpts do not contain it. An answer that supplies the fact anyway is
     incorrect, no matter how accurate.

Return ONLY valid JSON, no markdown fence:
{"grounded": true|false, "correct": true|false, "verdict": "pass"|"partial"|"fail", "why": "one short sentence"}

verdict: "pass" = grounded and correct. "partial" = right sources or right topic
but the answer is wrong or incomplete. "fail" = neither."""


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI. With 35 questions the interval is wide, and reporting a
    bare percentage would imply precision the sample size does not support."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_questions(smoke: bool, limit: int | None) -> list[dict]:
    questions = tomllib.loads(QUESTIONS.read_text(encoding="utf-8"))["question"]
    if smoke:
        questions = [q for q in questions if q["priority"] >= 5]
    return questions[:limit] if limit else questions


async def ask(client: httpx.AsyncClient, question: dict,
              retriever: str | None, top_k: int | None) -> dict:
    payload: dict = {"question": question["text"]}
    if retriever:
        payload["retriever"] = retriever
    if top_k:
        payload["top_k"] = top_k
    response = await client.post(f"{API_URL}/ask", json=payload)
    if response.status_code == 404:
        # No chunk passed the threshold. For an unanswerable question that is a
        # legitimate way to be right, so it is recorded rather than treated as
        # an error.
        return {"declined_by_retrieval": True, "sources": [], "answer": ""}
    response.raise_for_status()
    return response.json()


def score_retrieval(question: dict, sources: list[dict]) -> dict:
    """recall@k and MRR against expected_post_ids."""
    expected = set(question.get("expected_post_ids") or [])
    if not expected:
        return {"applicable": False}

    retrieved = [s["post_id"] for s in sources]
    first_rank = next((i for i, pid in enumerate(retrieved, 1) if pid in expected), None)
    found = expected & set(retrieved)
    return {
        "applicable": True,
        "hit": bool(found),
        "rank": first_rank,
        "rr": 1.0 / first_rank if first_rank else 0.0,
        "coverage": len(found) / len(expected),   # matters for multi_source
    }


async def judge(question: dict, result: dict) -> dict:
    excerpts = "\n\n".join(
        f"[{s['n']}] {s['title']}\n{s['text'][:900]}" for s in result.get("sources", [])
    ) or "(no excerpts were retrieved)"
    answer = result.get("answer") or "(the system returned no answer)"

    response = await llm.call(
        [{"role": "user", "content": JUDGE_PROMPT % (
            question["text"], excerpts, answer, question.get("grading_notes", ""))}],
        role="eval_judge", model=JUDGE_MODEL,
    )
    raw = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        logger.warning("%s: judge returned non-JSON", question["id"])
        return {"verdict": "error", "grounded": False, "correct": False,
                "why": "judge returned unparseable output", "cost_usd": response.usage.cost_usd}
    save_judgement(key, {k: v for k, v in parsed.items() if k != "cost_usd"})
    parsed["cost_usd"] = response.usage.cost_usd
    parsed["cached"] = False
    return parsed


def check_assertions(question: dict, answer: str) -> dict:
    must = [s for s in question.get("answer_must_contain", []) if s not in answer]
    must_not = [s for s in question.get("answer_must_not_contain", []) if s in answer]
    return {"missing": must, "forbidden_present": must_not,
            "passed": not must and not must_not}


async def run_one(client: httpx.AsyncClient, question: dict, retriever: str | None,
                  top_k: int | None, use_judge: bool, cache: dict[str, dict]) -> dict:
    try:
        result = await ask(client, question, retriever, top_k)
    except httpx.HTTPError as exc:
        logger.error("%s: API error %s", question["id"], exc)
        return {"id": question["id"], "error": str(exc)}

    row = {
        "id": question["id"],
        "type": question["type"],
        "priority": question["priority"],
        "holdout": question["holdout"],
        "retrieval": score_retrieval(question, result.get("sources", [])),
        "assertions": check_assertions(question, result.get("answer", "")),
        "n_sources": len(result.get("sources", [])),
        "declined_by_retrieval": result.get("declined_by_retrieval", False),
        "answer": result.get("answer", ""),
        "_sources": result.get("sources", []),
        "cost_usd": result.get("usage", {}).get("cost_usd", 0.0),
    }
    if use_judge:
        verdict = await judge(question, result, cache)
        row["judge"] = verdict
        row["cost_usd"] += verdict.pop("cost_usd", 0.0)
    return row


def report(rows: list[dict], meta: dict) -> None:
    print()
    print("=" * 74)
    print(f"EVAL  {meta['run_id']}   retriever={meta['retriever']}  n={len(rows)}")
    print("=" * 74)

    # --- retrieval ---
    scored = [r for r in rows if r["retrieval"].get("applicable")]
    if scored:
        hits = sum(1 for r in scored if r["retrieval"]["hit"])
        lo, hi = wilson(hits, len(scored))
        mrr = statistics.mean(r["retrieval"]["rr"] for r in scored)
        print(f"\nRETRIEVAL  (n={len(scored)}, questions with known sources)")
        print(f"  recall@k  {hits}/{len(scored)} = {hits/len(scored):.1%}"
              f"   95% CI [{lo:.1%}, {hi:.1%}]")
        print(f"  MRR       {mrr:.3f}")
        ranks = [r["retrieval"]["rank"] for r in scored if r["retrieval"]["rank"]]
        if ranks:
            print(f"  rank of first correct: median {statistics.median(ranks):.0f}, "
                  f"worst {max(ranks)}")
        misses = [r["id"] for r in scored if not r["retrieval"]["hit"]]
        if misses:
            print(f"  MISSED: {', '.join(misses[:6])}")

    # --- grounding ---
    judged = [r for r in rows if "judge" in r]
    if judged:
        verdicts = Counter(r["judge"].get("verdict") for r in judged)
        passes = verdicts.get("pass", 0)
        lo, hi = wilson(passes, len(judged))
        grounded = sum(1 for r in judged if r["judge"].get("grounded"))
        print(f"\nGROUNDING  (n={len(judged)}, LLM judge: {JUDGE_MODEL})")
        print(f"  pass      {passes}/{len(judged)} = {passes/len(judged):.1%}"
              f"   95% CI [{lo:.1%}, {hi:.1%}]")
        print(f"  verdicts  {dict(verdicts)}")
        print(f"  grounded  {grounded}/{len(judged)} = {grounded/len(judged):.1%}"
              "   (no claims outside the excerpts)")

    # --- by bucket ---
    print("\nBY TYPE")
    for qtype in sorted({r["type"] for r in rows}):
        subset = [r for r in rows if r["type"] == qtype]
        judged_sub = [r for r in subset if "judge" in r]
        passes = sum(1 for r in judged_sub if r["judge"].get("verdict") == "pass")
        line = f"  {qtype:<13} n={len(subset):<3}"
        if judged_sub:
            line += f" pass {passes}/{len(judged_sub)}"
        applicable = [r for r in subset if r["retrieval"].get("applicable")]
        if applicable:
            hit = sum(1 for r in applicable if r["retrieval"]["hit"])
            line += f"  recall {hit}/{len(applicable)}"
        print(line)

    # --- holdout, the eval-overfitting check ---
    for label, flag in (("tuned", False), ("HELD OUT", True)):
        subset = [r for r in rows if r["holdout"] is flag and "judge" in r]
        if subset:
            passes = sum(1 for r in subset if r["judge"].get("verdict") == "pass")
            print(f"  {label:<13} n={len(subset):<3} pass {passes}/{len(subset)}")

    failures = [r for r in rows if r.get("judge", {}).get("verdict") in ("fail", "partial")]
    if failures:
        print("\nFAILURES")
        for row in failures[:8]:
            print(f"  [{row['judge']['verdict']:<7}] {row['id']}")
            print(f"            {row['judge'].get('why', '')[:88]}")

    print(f"\ncost this run: ${sum(r.get('cost_usd', 0) for r in rows):.4f}")
    print("=" * 74)


async def main_async(args) -> None:
    questions = load_questions(args.smoke, args.limit)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger.info("Running %d questions (run_id=%s)", len(questions), run_id)

    async with httpx.AsyncClient(timeout=180) as client:
        try:
            health = (await client.get(f"{API_URL}/health")).json()
        except httpx.HTTPError as exc:
            logger.error("API unreachable at %s (%s). Start it first.", API_URL, exc)
            sys.exit(1)
        logger.info("API: retriever=%s model=%s", health["retriever"], health["model"])

        cache = {} if args.no_cache else load_judge_cache()
        if cache:
            logger.info("Judge cache: %d prior judgements", len(cache))
        semaphore = asyncio.Semaphore(args.concurrency)

        async def guarded(q):
            async with semaphore:
                return await run_one(client, q, args.retriever, args.top_k,
                                     not args.no_judge, cache)

        rows = await asyncio.gather(*(guarded(q) for q in questions))

    meta = {
        "run_id": run_id,
        "retriever": args.retriever or health["retriever"],
        "model": health["model"],
        "embedding_model": health.get("embedding_model"),
        "max_distance": health.get("max_distance"),
        "judge": None if args.no_judge else JUDGE_MODEL,
        "n": len(rows),
    }

    # One append-only file, one row per case per run -- deliberately NOT a
    # directory of timestamped dumps (see _knowledge/02: that pattern made
    # run-over-run comparison manual in the project we borrowed this design from).
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "run": meta}, ensure_ascii=False) + "\n")

    if args.dump:
        detail = [{
            "id": q["id"], "type": q["type"], "priority": q["priority"],
            "holdout": q["holdout"], "question": q["text"],
            "grading_notes": q.get("grading_notes", ""),
            "expected_post_ids": q.get("expected_post_ids", []),
            "answer": r.get("answer", ""),
            "retrieval": r.get("retrieval", {}),
            "sources": [
                {"n": s["n"], "post_id": s["post_id"], "title": s["title"],
                 "score": s["score"], "text": s["text"][:700]}
                for s in (r.get("_sources") or [])[:5]
            ],
        } for q, r in zip(questions, rows)]
        Path(args.dump).write_text(
            json.dumps(detail, ensure_ascii=False, indent=1), encoding="utf-8")
        logger.info("Wrote grading detail for %d questions to %s", len(detail), args.dump)

    report(rows, meta)
    logger.info("Appended %d rows to %s", len(rows), RESULTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG eval set")
    parser.add_argument("--smoke", action="store_true", help="priority 5 only")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retriever", choices=["dense", "bm25"])
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--no-judge", action="store_true",
                        help="retrieval metrics only; no LLM judge calls")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--dump", metavar="PATH",
                        help="Write question/answer/sources to PATH for manual grading")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cached judgements and re-grade everything")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
