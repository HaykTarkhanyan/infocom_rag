# Code Audit

## Bug

### `rag.py` ignores `WEAVIATE_URL` config — always connects to localhost

`rag.py:33`: `weaviate.connect_to_local()` with no args, while `ingest.py:320-325` correctly parses `WEAVIATE_URL`. If someone sets `WEAVIATE_URL=http://remote-host:9090`, ingestion hits the right server but all queries silently go to `localhost:8080`.

## Dead Code

1. **`EMBEDDING_DIM`** (`config.py:15`) — defined but never imported or used anywhere in the codebase.

2. **`HNSW_EF`** (`config.py:51`) — imported in `rag.py:18` but never passed to any query or config. It's also missing from `create_collection` in `ingest.py:233`, where only `ef_construction` and `max_connections` are set. The query-time `ef` parameter advertised in the docs/config is completely inert.

## Design Issues

1. **Blocking sync call in async event loop** (`bot.py:44`) — `rag.answer(query)` is synchronous (embedding + Weaviate + Gemini HTTP calls). Calling it directly in the `async def handle_message` blocks the entire event loop. The bot freezes for *all* users while one query processes. Should use `asyncio.to_thread(rag.answer, query)`.

2. **`RAG` doesn't implement context manager** (`rag.py:22`) — Has a `close()` method but no `__enter__`/`__exit__`. In `bot.py`, if `run_polling()` raises, the Weaviate connection leaks. In `ui.py`, `close()` is never called at all.

3. **`create_collection` is silently destructive** (`ingest.py:224`) — Drops and recreates the collection every time with only a print. No `--force` flag, no confirmation, no way to do incremental ingestion. Running `ingest.py` twice wipes all prior data.

4. **`sys.exit(1)` inside a library function** (`scraper.py:54`) — `scrape_group` calls `sys.exit` on missing credentials, making it unusable as an importable function. Should raise an exception instead.

5. **Global mutable state** (`bot.py:23`) — `rag: RAG | None = None` as a module-level global with `global rag` in `main()`. Makes testing and reuse harder.

6. **No config validation** (`config.py`) — `int(os.getenv("CHUNK_WINDOW_SIZE", "5"))` will throw an opaque `ValueError` if someone puts `"abc"` in `.env`. No bounds checking either (e.g., negative `TOP_K`).

## Inefficiencies

1. **No GPU auto-detection** (`embeddings.py:19`) — Model is always on CPU. Adding `device = "cuda" if torch.cuda.is_available() else "cpu"` and `.to(device)` is a trivial fix for a major throughput difference.

2. **No internal batching in `embed()`** (`embeddings.py:23`) — If called with many texts (via `embed_documents`), all are tokenized and run through the model in one pass. For 64 long texts (the default `INGEST_BATCH_SIZE`), this can OOM on limited hardware. Should chunk internally.

3. **Import inside nested function** (`ingest.py:148`) — `from collections import deque` is imported inside `collect_thread()`, which runs once per root message. Should be a top-level import.

## Inconsistencies

1. **Weaviate connection style differs between modules** — `ingest.py` parses `WEAVIATE_URL` into host/port and calls `connect_to_local(host=..., port=...)`. `rag.py` calls `connect_to_local()` bare. Two different patterns for the same operation.

2. **Scraper produces numeric sender IDs, ingest expects human-readable names** — `scraper.py:138`: `to_telegram_export_format` does `str(m.get("sender_id", "Unknown"))`, producing `"123456"` as the sender. But Telegram Desktop exports use display names like `"Alice"`. The scraper fetches `message.sender_id` but never resolves it to a display name, so scraped-then-ingested data will have opaque numeric senders in the search results.

3. **Hardcoded chat name in converter** — `scraper.py:129` hardcodes `"name": "infocomm"` while `scraper.py:36` has `DEFAULT_GROUP = "infocomm"` as a separate constant. The `--group` arg is not passed through to the conversion.

4. **`date` stored as TEXT in Weaviate** (`ingest.py:242`) — Prevents date-range filtering or sorting at the DB level. Should be `DataType.DATE`.

5. **`return_metadata=["distance"]`** (`rag.py:52`) — Weaviate v4 client expects `return_metadata=wvc.query.MetadataQuery(distance=True)`, not a string list. This may work by coincidence in some client versions but is not the documented API.

## Robustness Gaps

1. **No prompt injection mitigation** (`rag.py:87-92`) — Chat messages are inserted verbatim into the Gemini prompt as "context." A malicious message in the Telegram history like `"Ignore all previous instructions and..."` gets passed straight through. Consider at minimum a delimiter/escaping strategy.

2. **No retry on Gemini API errors** (`rag.py:94`) — A transient Gemini 429/500 crashes the entire request with no retry. The bot handler catches this at `bot.py:46`, but the user just sees a generic error.

3. **No timeout on Weaviate or Gemini calls** — If either service hangs, the bot/UI hangs indefinitely.

4. **`requirements.txt` uses `>=` only** — No upper bounds or lockfile. A breaking release of any dependency (weaviate-client v5, transformers, etc.) could silently break the project.

## Minor

- `scraper.py:99`: bare `except Exception` swallows all download errors with just a print. Could lose important errors silently.
- `ui.py:37`: Gemini model list is hardcoded in the UI rather than derived from config or an API call.
- Tests repeat the env setup boilerplate (`os.environ.setdefault(...)`) identically across all 5 test files — a shared `conftest.py` would be cleaner.
- `test_ingest.py` / `test_rag.py` use `sys.modules.setdefault()` to mock heavy deps. This is fragile and order-dependent — a `conftest.py` with proper fixtures would be more maintainable.
