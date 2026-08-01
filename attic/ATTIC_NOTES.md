# Attic — archived first attempt (2026-08-01)

Everything in this folder is the original Telegram-chat RAG prototype, archived when the
project restarted as an infocom.am news RAG. Reference only. Nothing here is imported by
the new code. Read it, steal from it, don't revive it wholesale.

(`README.md` and `REPORT.md` in this folder are the *old* project's docs, not the new
project's. The new README lives at the repo root.)

Why it was retired: it was designed for Telegram chat messages and later pointed at a
195k-article news corpus without redesigning anything below the scraper. The schema
(`TelegramMessages`, `sender`, `reply_to_message_id`), all three chunking strategies, and
the system prompt were all chat-shaped.

## Worth salvaging

- **`web_scraper.py`** — the only part that ran for real against infocom.am. The WP REST
  API endpoints, pagination via `X-WP-Total`, the `infotag` custom taxonomy, and the
  HTML-to-text regexes are all correct and worth lifting. Its weak spot is direct dict
  indexing on the author/category lookups (`author_map[author_id]`), which hard-crashes
  on any API-hidden author.
- **`embeddings.py`** — the e5 `query:`/`passage:` prefix convention and mean-pooling +
  L2 normalize are right for `Metric-AI/armenian-text-embeddings-1`. Reuse the approach,
  not the file (no batching, no device selection).
- **`articles.jsonl`** — 5 real scraped articles, useful as a test fixture.

## Known defects (do not reintroduce)

1. **Silent truncation.** 512-token limit with `truncation=True`, one chunk per whole
   article. Measured on the sample: an 835-word Armenian article is 2,340 tokens, so 78%
   of it never reached the vector while the full text was still sent to the LLM. Armenian
   runs ~2.2-2.8 tokens/word through XLM-R, so 512 tokens is only ~185-230 words.
2. **`ingest.py` dropped and recreated the collection on every run.** No incremental
   upsert, no deterministic IDs.
3. **`MAX_DISTANCE=1.0`** on cosine distance filters essentially nothing (e5 similarities
   cluster in 0.7-0.95).
4. **`HNSW_EF`** was imported and never used. `date` was stored as TEXT, so no date
   filtering — the most valuable filter on a news corpus.
5. **`bot.py` blocked its own event loop** — sync `rag.answer()` inside `async def`.
6. **No evaluation harness**, despite a REPORT.md section on tuning for quality and a UI
   full of sliders.

## About `AUDIT.md`

Mostly accurate, with one error: it claims `return_metadata=["distance"]` is not the
documented Weaviate v4 API. It is — `METADATA` is typed as
`Union[List[Literal['distance', ...]], MetadataQuery]` in weaviate-client 4.9. The audit
also never questioned whether the design fit the data, which was the actual problem.

## Test suite

71 tests passed, and they were close to worthless for confidence: every one mocked
`torch`, `transformers`, `weaviate`, and `google.genai` at the `sys.modules` level, so
nothing numerical or I/O-related was ever exercised. The RAG half of the pipeline never
ran against a real vector or a live Weaviate.
