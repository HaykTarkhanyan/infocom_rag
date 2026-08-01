# research/ — investigation and dev tooling (not part of the pipeline)

This is the **non-essential tier** for this repo. Nothing here runs as part of the
pipeline; `src/` holds everything load-bearing:

| tier | location | contents |
|---|---|---|
| load-bearing | `src/` | `fetch_articles.py`, `chunking.py`, `download_model.py` — run these to get a working corpus |
| non-essential | `research/` *(here)* | site investigation, extractor benchmark, tokenizer inspection |
| archive | *(none yet)* | nothing has become dead weight so far |

Every script here still runs and is worth re-running when the underlying thing
changes. Prose findings live in [`_knowledge/`](../_knowledge/README.md) — this
folder holds the tooling and its raw output (`raw/`), not the write-ups.

## Still-useful tools

- **`inspect_tokenizer.py`** — a local tiktokenizer for ATE-2: token counts,
  boundaries, and IDs for any text or any chunk in `data/chunks.jsonl`. Use this
  rather than a browser playground, which miscounts ATE-2 by 53%
  (see `_learnings/2026-08-01-2052_browser-tokenizers-miscount-ate2.md`).
  Re-run any time you need to reason about chunk budgets.

- **`compare_extractors.py`** — benchmarks regex vs selectolax vs trafilatura on
  real article bodies. Re-run if we ever reconsider the extractor choice, or if
  infocom.am changes its page-builder markup and extraction starts leaking junk.

## Finished investigations (re-run to refresh the facts)

These produced [`_knowledge/01_site_structure.md`](../_knowledge/01_site_structure.md). The site keeps publishing, so the numbers in
that document drift; re-run these to refresh rather than trusting the write-up
indefinitely.

- **`probe_api.py`** — WordPress REST API structure: post types, taxonomies,
  languages, authorship, custom namespaces, archive depth. Answers "what is
  actually available".

- **`probe_corpus.py`** — per-category content quality (word counts, bylines,
  empty bodies) and data-quality checks (placeholder dates, duplicate slugs,
  whether `excerpt` is a real summary). This is what identified `indepth` as the
  corpus worth indexing and `reels` as 30/30 empty.

- **`fetch_sitemaps.py`** — sitemap discovery. Mostly settled: `wp-sitemap.xml` is
  the real index (103 post shards), `sitemap_index.xml` 404s despite being
  advertised in robots.txt, and `news-sitemap.xml` is a rolling 114-URL window
  that would suit incremental polling.

## Notes

- Run everything from the repo root with `.venv/Scripts/python.exe`, and prefix
  with `PYTHONIOENCODING=utf-8` — Windows `cp1252` cannot print Armenian.
- `research/raw/category_samples.json` and `extractor_comparison.json` are
  gitignored (7.9 MB and 604 KB); regenerate them with the scripts above.
