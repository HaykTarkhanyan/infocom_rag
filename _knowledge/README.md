# Knowledge base — infocom.am RAG

Researched background and findings. Prose lives here; the scripts that produced
it live in [`research/`](../research/README.md), and raw artifacts in
`research/raw/`.

## Index

| # | File | Topic | What's inside |
|---|------|-------|---------------|
| 01 | [01_site_structure.md](01_site_structure.md) | infocom.am survey | Access surface (robots, sitemaps, WP REST API), URL patterns, category inventory with volumes, per-category content quality, data-quality problems, field mapping for ingestion |
| 02 | [02_evaluation_design.md](02_evaluation_design.md) | How to evaluate this RAG | Question-set schema, metrics, LLM-as-judge recipe, runner design, and the specific mistakes to avoid — surveyed from a prior text-to-SQL project |

## How this differs from `_learnings/`

- **`_knowledge/`** — researched background on a *topic*. Longer, numbered,
  revisited and updated as understanding deepens. "What is true about X."
- **`_learnings/`** — one gotcha or lesson per file, dated, discovered while
  working. "What bit us, and why."

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
