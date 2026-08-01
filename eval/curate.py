"""One-time bootstrap: turn validated candidates into eval/questions.toml.

Run ONCE. After that `eval/questions.toml` is hand-maintained -- an eval set is
curated, not regenerated, and re-running this would discard hand-written cases
(the unanswerable ones especially, which no generator can produce because they
are defined by absence from the corpus).

Filters applied, each with a reason:
  - `must_contain` strings that do not appear in the source chunk are dropped.
    The generator hallucinated 3 of 42; an eval that asserts on invented text is
    broken before it runs.
  - questions whose token overlap with their source exceeds MAX_OVERLAP are
    dropped. Those largely restate the passage and would flatter any retriever.

Ground truth is `expected_post_ids`, NOT chunk ids. Chunk ids are
`{post_id}-{section}-{part}` and change whenever chunking parameters change;
post ids are stable. "Did we retrieve the right article" is also the question
that actually matters.
"""

import json
import random
import re
from pathlib import Path

CANDIDATES = Path("eval/candidates.jsonl")
CHUNKS = Path("data/chunks.jsonl")
OUT = Path("eval/questions.toml")

MAX_OVERLAP = 0.65
HOLDOUT_FRACTION = 0.25
SEED = 509

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def overlap(question: str, source: str) -> float:
    q = set(TOKEN_RE.findall(question.lower()))
    return len(q & set(TOKEN_RE.findall(source.lower()))) / max(1, len(q))


def toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def toml_list(values: list[str]) -> str:
    return "[" + ", ".join(toml_str(v) for v in values) + "]"


def main() -> None:
    chunks = {c["chunk_id"]: c for c in
              (json.loads(l) for l in CHUNKS.open(encoding="utf-8") if l.strip())}
    candidates = [json.loads(l) for l in CANDIDATES.open(encoding="utf-8") if l.strip()]

    kept = []
    dropped_mc = dropped_ov = 0
    for cand in candidates:
        source = chunks.get(cand["source_chunk_id"])
        if source is None:
            continue
        if any(m not in source["text"] for m in cand["must_contain"]):
            dropped_mc += 1
            continue
        ov = overlap(cand["text"], source["text"])
        if ov > MAX_OVERLAP:
            dropped_ov += 1
            continue
        cand["overlap"] = ov
        kept.append(cand)

    rng = random.Random(SEED)
    rng.shuffle(kept)
    n_holdout = int(len(kept) * HOLDOUT_FRACTION)

    lines = [
        "# Evaluation question set.",
        "#",
        "# HAND-MAINTAINED. eval/curate.py bootstrapped the generated cases once;",
        "# do not re-run it, it would discard the hand-written ones.",
        "#",
        "# Ground truth is `expected_post_ids` (stable) rather than chunk ids",
        "# (which change whenever chunking parameters change).",
        "#",
        "# holdout = true marks cases that must NEVER be tuned against. Report them",
        "# separately; if held-out accuracy drifts below the tuned set, the tuning",
        "# has started fitting the eval rather than improving the system.",
        "#",
        "# priority: 5 = must pass, 1 = nice to have. Used to run a subset when short",
        "# on time or budget.",
        "#",
        "# type:",
        "#   factual      answerable from one article",
        "#   multi_source needs more than one article",
        "#   unanswerable NOT in the corpus -- the system must decline, not invent.",
        "#                expected_post_ids is empty and answer_must_contain is a",
        "#                refusal marker.",
        "",
    ]

    for i, cand in enumerate(kept):
        holdout = i < n_holdout
        priority = 4 if cand["kind"] == "specific" else 3
        slug = re.sub(r"[^a-z0-9]+", "-",
                      cand["source_title"].lower()[:28]).strip("-") or "q"
        lines += [
            "[[question]]",
            f'id = {toml_str(f"{slug}-{cand['source_post_id']}-{cand['kind'][:4]}")}',
            f"priority = {priority}",
            'type = "factual"',
            f'text = {toml_str(cand["text"])}',
            f"expected_post_ids = [{cand['source_post_id']}]",
            f"answer_must_contain = {toml_list(cand['must_contain'])}",
            "answer_must_not_contain = []",
            f'grading_notes = {toml_str("Must answer from the cited article and include the exact figures or names listed in answer_must_contain.")}',
            f"holdout = {str(holdout).lower()}",
            f"source_overlap = {cand['overlap']:.2f}   # question/source token overlap",
            f'source_url = {toml_str(cand["source_url"])}',
            "",
        ]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"kept {len(kept)} of {len(candidates)} candidates")
    print(f"  dropped, hallucinated must_contain : {dropped_mc}")
    print(f"  dropped, overlap > {MAX_OVERLAP}          : {dropped_ov}")
    print(f"  marked holdout                     : {n_holdout}")
    print(f"wrote {OUT}")
    print()
    print("NEXT: hand-add unanswerable and multi_source cases. A set with no")
    print("unanswerable cases cannot detect hallucination, which is the failure")
    print("mode that matters most.")


if __name__ == "__main__":
    main()
