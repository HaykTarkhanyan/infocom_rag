# Knowledge base — infocom.am RAG

Researched background and findings. Prose lives here; the scripts that produced
it live in [`research/`](../research/README.md), and raw artifacts in
`research/raw/`.

## Index

| # | File | Topic | What's inside |
|---|------|-------|---------------|
| 01 | [01_site_structure.md](01_site_structure.md) | infocom.am survey | Access surface (robots, sitemaps, WP REST API), URL patterns, category inventory with volumes, per-category content quality, data-quality problems, field mapping for ingestion |
| 02 | [02_evaluation_design.md](02_evaluation_design.md) | How to evaluate this RAG | Question-set schema, metrics, LLM-as-judge recipe, runner design, and the specific mistakes to avoid — surveyed from a prior text-to-SQL project |
| 03 | [03_armenian_llm_benchmarks.md](03_armenian_llm_benchmarks.md) | ArmBench-LLM 1.0 | Which LLM to pick for Armenian, read by reading-comprehension score rather than the headline Average; cost nuances; why several leaderboard zeros are harness failures |

## How this differs from the other note folders

- **`_knowledge/`** — researched background on a *topic*. Longer, numbered,
  revisited and updated as understanding deepens. "What is true about X."
- **`_learnings/`** — one gotcha or lesson per file, dated, discovered while
  working. "What bit us, and why."
- **`DECISIONS.md`** (repo root) — the choices themselves, with alternatives
  rejected and what would reopen them. "Why is it built this way."

A finding that took an afternoon of investigation goes here; the choice it led to
goes in `DECISIONS.md` with a link back.

A finding that took an afternoon of investigation goes here. A trap that cost an
hour of debugging goes in `_learnings/`.

## Current state of play

The pipeline fetches and chunks; it does not yet embed, store, or retrieve.

Settled so far:
- Corpus target is `indepth` (category 51) — 5,836 long-form authored articles,
  not the 146,936 wire stubs in `news`. See 01.
- Chunking is heading-aware and token-budgeted against ATE-2's real tokenizer.
  94 articles → 969 chunks, none over 512 tokens, no content dropped.
- Extraction is selectolax, chosen by measurement rather than benchmark
  reputation.

Open, in rough dependency order:
1. Embedding — weights not downloaded yet (bandwidth), base vs large undecided.
2. Vector store — not chosen. Hybrid retrieval (BM25 + dense) is the design
   default per 02.
3. **Evaluation — nothing exists.** Per 02 this should come before any retrieval
   tuning, or every knob is guesswork.

## Provenance

- `01_site_structure.md` — first-hand, 2026-08-01, via Playwright + the WP REST
  API. Reproducible with `research/probe_api.py` and `research/probe_corpus.py`.
  The site publishes daily, so its counts drift; re-run rather than trusting them
  indefinitely.
- `02_evaluation_design.md` — second-hand, 2026-08-01, surveyed from
  `Desktop/metric/Washington` by a subagent. Claims about that repo have not been
  independently verified file-by-file; treat the file paths as pointers to go
  read, not as established fact.
- `03_armenian_llm_benchmarks.md` — read first-hand, 2026-08-01, from the
  verbatim blog and leaderboard snapshots in
  `Desktop/metric/ArmBench-LLM/references/` (captured 2026-06-15, published
  2026-04-02). The live leaderboard moves and the newest Gemini models are not
  on it, so re-check before treating the ranking as current.
