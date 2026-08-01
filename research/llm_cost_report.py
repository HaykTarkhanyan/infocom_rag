"""Summarize the LLM call ledger: tokens, cost, latency.

Reads the append-only JSONL written by src/llm.py (one object per call).

Usage:
    python research/llm_cost_report.py
    python research/llm_cost_report.py --by model
    python research/llm_cost_report.py --by day --since 2026-08-01
    python research/llm_cost_report.py --tail 10
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import settings


def load(path: Path, since: str | None) -> list[dict]:
    if not path.exists():
        print(f"No ledger at {path} yet -- nothing has called the LLM.")
        sys.exit(0)
    rows = []
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"WARNING: {path}:{line_no} is not valid JSON ({exc}); skipping")
            continue
        if since and row.get("ts", "") < since:
            continue
        rows.append(row)
    return rows


def summarize(rows: list[dict], key: str) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if key == "day":
            label = row.get("ts", "")[:10]
        else:
            label = str(row.get(key, "?"))
        groups[label].append(row)

    header = f"{key:<38} {'calls':>6} {'in':>10} {'out':>9} {'cost $':>10} {'p50 ms':>8}"
    print(header)
    print("-" * len(header))
    for label in sorted(groups):
        batch = groups[label]
        latencies = [r.get("latency_ms", 0) for r in batch]
        print(
            f"{label[:38]:<38} {len(batch):>6} "
            f"{sum(r.get('input_tokens', 0) for r in batch):>10,} "
            f"{sum(r.get('output_tokens', 0) for r in batch):>9,} "
            f"{sum(r.get('cost_usd', 0.0) for r in batch):>10.5f} "
            f"{int(statistics.median(latencies)) if latencies else 0:>8,}"
        )


def totals(rows: list[dict]) -> None:
    if not rows:
        print("No calls recorded in range.")
        return
    cost = sum(r.get("cost_usd", 0.0) for r in rows)
    tin = sum(r.get("input_tokens", 0) for r in rows)
    tout = sum(r.get("output_tokens", 0) for r in rows)
    cached = sum(r.get("cached_tokens", 0) for r in rows)
    latencies = [r.get("latency_ms", 0) for r in rows]
    latencies.sort()

    print()
    print("=" * 64)
    print(f"  calls          {len(rows):,}")
    print(f"  input tokens   {tin:,}" + (f"  ({cached:,} cached)" if cached else ""))
    print(f"  output tokens  {tout:,}")
    print(f"  TOTAL COST     ${cost:.5f}")
    if rows:
        print(f"  cost per call  ${cost / len(rows):.6f}")
    if latencies:
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        print(f"  latency        p50 {int(statistics.median(latencies)):,} ms   p95 {p95:,} ms")
    print(f"  window         {rows[0].get('ts', '?')} .. {rows[-1].get('ts', '?')}")
    print("=" * 64)


def tail(rows: list[dict], n: int) -> None:
    print(f"{'ts':<21} {'role':<12} {'model':<34} {'in':>7} {'out':>6} {'cost $':>9}")
    print("-" * 94)
    for row in rows[-n:]:
        print(
            f"{row.get('ts', '?'):<21} {str(row.get('role', '?'))[:12]:<12} "
            f"{str(row.get('model', '?'))[:34]:<34} "
            f"{row.get('input_tokens', 0):>7,} {row.get('output_tokens', 0):>6,} "
            f"{row.get('cost_usd', 0.0):>9.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the LLM cost ledger")
    parser.add_argument("--ledger", default=settings.logging.llm_ledger)
    parser.add_argument("--by", choices=["model", "role", "day"], default="model")
    parser.add_argument("--since", help="ISO date/time lower bound, e.g. 2026-08-01")
    parser.add_argument("--tail", type=int, help="Show the last N calls instead")
    args = parser.parse_args()

    rows = load(Path(args.ledger), args.since)
    if args.tail:
        tail(rows, args.tail)
    else:
        summarize(rows, args.by)
    totals(rows)


if __name__ == "__main__":
    main()
