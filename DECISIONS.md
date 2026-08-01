# Design decisions

Newest first. Deep supporting research lives in [`_knowledge/`](_knowledge/README.md);
this file holds the choice and a pointer.

Entries 1-11 were written on 2026-08-01 covering decisions made earlier the same
day, before this file existed. Later entries are recorded as the decision is made.

---

## 17. The API and LLM client are async, converted together

**Date** 2026-08-01 · **Status** active · **Driver:** concurrent users are expected

**Why.** A request spends ~5 seconds waiting on OpenRouter. Sync handlers were
*correct* — FastAPI runs `def` endpoints in a threadpool, verified at 3
concurrent requests — but each one holds an OS thread for the whole wait, and
anyio's default limit is **40**. That caps sustained throughput near 8 req/s and
queues everything after. Async makes the wait cost a coroutine instead.

**Converted as one unit, deliberately.** `async def` handlers around a blocking
`requests.post` would be *worse* than staying sync: it parks the whole event loop
for 5 seconds per request. That is precisely the archived prototype's bug (sync
`rag.answer()` inside `async def handle_message`). So the client moved to
`httpx.AsyncClient` in the same change, and it is now shared process-wide — which
also removes the TCP + TLS handshake that `requests` was paying on every call.

BM25 scoring is CPU-bound, so it goes through `asyncio.to_thread` rather than
blocking the loop. The index is warmed in the lifespan, so no unlucky first
request pays to build it and concurrent requests cannot race to build it at once.

**Verified:** 6 concurrent requests, 50.67s of work in 9.48s wall clock, all 200.

**Fixed alongside — a live data-loss bug.** The cost ledger did an unguarded
`open(path, "a")` per call and lost **~10% of rows** under 8 threads, with zero
corrupt lines. It was already happening, since sync handlers were already
concurrent. Now guarded by a `threading.Lock`, with a 200-write regression test.

**Still open for real concurrent users.** Authentication, per-user cost caps and
rate limiting do not exist. A shared API key with no ceiling is an unbounded
spend risk the moment the UI is reachable by anyone else. Also, the ledger lock
is per-process — multiple uvicorn workers would race again, and at that point
cost accounting should move to Postgres.

**What would change this.** Multi-worker deployment (move the ledger to the DB),
or streaming responses (would want SSE through the API to the UI).

---

## 16. Retrieval is BM25 for now, explicitly as a stopgap

**Date** 2026-08-01 · **Status** active, **temporary by design**

**Why.** Dense retrieval is blocked on downloading ATE-2 weights. BM25 over the
969 chunks works today with no model and no vector store, and it is not throwaway:
the intended design is hybrid (BM25 + dense), so this is the first half.

**Known weakness, deliberately not patched.** Armenian is agglutinative and there
is no stemmer, so `ժողովը` and `ժողով` are different terms. Dense retrieval is
the fix; hand-written suffix rules would be a worse one.

**Also visible already:** with `top_k = 10` the tail of the results is junk — a
question about road-accident data returns articles about flower exports and
scientist attestation. A `min_score` cutoff exists in the API but has no
principled default, because BM25 scores are corpus-relative. The eval set should
set it.

**What would change this.** Weights downloaded → dense retrieval → hybrid. The
whole surface is `search(query, top_k) -> list[Hit]`, so the swap touches
`src/retrieval.py` alone.

---

## 15. Chainlit's data layer replaces our own persistence schema

**Date** 2026-08-01 · **Status** active (supersedes #13)

**Why.** Chainlit persists users, threads, steps, elements and feedback itself.
Adopting its schema buys chat history, thread resume and the feedback UI for
free; maintaining a parallel schema would mean writing every row twice.

**What it costs.** The shape is Chainlit's, not ours. `config_snapshot`,
`retrieved`-with-scores and per-turn cost have no columns of their own.
**Mitigated:** the app writes all of it into `steps.metadata` (JSONB), so
per-answer accounting survives — `src/db.py --check` sums cost straight out of
`metadata->>'cost_usd'`.

**Alternatives rejected.** Keeping both schemas (double writes, two sources of
truth); keeping ours and forgoing Chainlit's history/resume/feedback UI.

**Trap, cost half a debugging session.** Chainlit's *published* DDL is stale
against the shipped package — 2.11.1 writes an `autoCollapse` column the docs
never create — and step persistence is fire-and-forget, so every insert failed
while the UI looked perfect. Derive the schema from `chainlit.step.StepDict`, and
verify rows actually land before believing it works. See
[`_learnings/2026-08-01-2205_chainlit-schema-stale-and-silent-persistence.md`](_learnings/2026-08-01-2205_chainlit-schema-stale-and-silent-persistence.md).

**What would change this.** Outgrowing Chainlit, or needing turn-level fields
that do not fit in step metadata.

---

## 14. UI is Chainlit over a FastAPI `/ask` endpoint

**Date** 2026-08-01 · **Status** active

**Why.** Chainlit is purpose-built for chat and `cl.Step` renders the pipeline as
a collapsible tree natively — retrieval with per-chunk scores, then generation
with the assembled prompt, tokens and cost. That *is* the debug view, with no
custom layout. Streamlit would need it hand-built from expanders, and its
rerun-everything model fights chat.

FastAPI sits underneath as the only entry point. The UI and the eval harness both
call `/ask` over HTTP rather than importing the pipeline, so the eval exercises
the same code path a user does — scripts that import internals drift and then
pass while the real path is broken.

**Alternatives rejected.**
- **Streamlit** — better for non-chat dashboards (corpus stats, eval tables), and
  still the right choice if those appear. Worse for this.
- **Open WebUI** — zero UI code and a polished product, but it treats a custom
  backend as a black-box chat model and *cannot* display sources or scores
  structurally; citations would have to be markdown baked into the answer text.
  That deletes the debug half. Its licence also changed in April 2025 (branding
  clause, binding only above 50 users). Still the best option later if a
  polished demo matters more than introspection.
- **Chainlit's maintenance risk** was weighed and found overstated: the founders
  stepped back in May 2025, but it is maintained under a formal Maintainer
  Agreement and shipped v2.11.1 in April 2026.

**What would change this.** Wanting dashboards alongside chat (add Streamlit
beside it, both on the same API), or wanting a public polished demo (add Open
WebUI against an OpenAI-compatible shim).

---

## 13. Sessions, turns and feedback persist to Neon Postgres

**Date** 2026-08-01 · **Status** **superseded by #15**

**Why.** Three tables, mirroring the shape the Washington project settled on and
adapted from text-to-SQL to document retrieval. The JSONL ledger
(`logs/llm_calls.jsonl`) records *calls* and is per-machine and append-only;
Postgres records *turns* — question, answer, which chunks were retrieved, and the
feedback on them — which is what makes quality reviewable rather than just
countable.

Choices inside the schema, each deliberate:
- **`cost_usd` is `NUMERIC(12,6)`, not float.** Costs are thousands of
  ~$0.002 charges; binary float accumulates error across that many additions.
  Verified it reads back as `Decimal('0.002152')`.
- **`retrieved` is a JSONB snapshot**, not foreign keys into a chunk table. The
  corpus gets re-chunked, and history has to replay what was actually used, not
  what a chunk id points at today.
- **`config_snapshot` per turn.** Records the model, temperature and retrieval
  settings that produced the answer. Without it a stored answer cannot be
  reproduced or fairly compared against a later run — which matters precisely
  because those settings are now pinned and will change.
- **Timestamps are `TIMESTAMPTZ`**, never naive.
- Identity is `(session_id, turn_idx)`; feedback cascades from turns, turns from
  sessions.

**Alternatives rejected.** SQLite (no shared access, and Neon is already
available); the JSONL ledger alone (no join between an answer and the feedback on
it, and no place for retrieval detail); a chunk foreign key (breaks on
re-chunking).

**Deliberately not built yet.** Eval-run tables. `_knowledge/02` recommends a
compact results store instead of loose JSON files, and Postgres is the obvious
home — but no eval harness exists, so the schema would be guesswork.

**What would change this.** Needing concurrent writers (would want a pool rather
than a connection per call), or eval results outgrowing files.

---

## 12. Generation model is `openai/gpt-5.4-mini`

**Date** 2026-08-01 · **Status** active (supersedes #11)

**Why.** Two independent measurements both favour it, and the second was a
surprise.

1. **Reading comprehension.** This is a RAG system, so the model never recalls
   facts — it reads supplied passages. On the ArmBench-LLM reading sub-scores
   (not the knowledge-weighted Average), gpt-5.4-mini scores 0.965 against 0.827
   for the best Gemini available on OpenRouter.
2. **Armenian tokenizer efficiency.** Measured on five real chunks of our corpus
   (784 words), gpt-5.4-mini used 1,952 prompt tokens (2.49 tok/word) against
   Gemini's 3,189 (4.07 tok/word) — **63% more tokens for identical text**. That
   cancels Gemini's lower headline rate: the same input cost $0.001486 via
   gpt-5.4-mini and $0.001597 via gemini-3-flash-preview.

So the model that reads Armenian better is also cheaper on Armenian, despite
listing at a 50% higher per-token rate. Entry #11 assumed Gemini was the cheaper
option; on a per-token basis that was true and on a per-Armenian-word basis it
was wrong.

**Verified live**, not just on paper: a real end-to-end call answered an Armenian
question over four real chunks, in Armenian, with a correct `[3]` citation, and
declined to invent limitations the excerpts did not contain. ~$0.002 per query at
realistic context size.

**Also settled by testing:** `openai/gpt-5.*` models are commonly reasoning
models that reject a custom `temperature`, which would have broken our client.
gpt-5.4-mini **accepts** `temperature=0.0` — confirmed by a live call — so no
special-casing was needed.

**Alternatives rejected.**
- `google/gemini-3-flash-preview` (#11) — worse reading, worse Armenian
  tokenization, kept as the cross-vendor fallback.
- `openai/gpt-5.2-pro` — best reading (0.973) but roughly 50x the cost.
- `openai/gpt-5.4-nano` — cheapest, reading 0.877; a real option if cost ever
  matters, which at ~$2 per 1,000 questions it currently does not.
- `openai/flex` provider tier — half price ($0.375/$2.25) for best-effort
  latency. Not the default, but noted in `config.toml` as worthwhile for eval
  sweeps where latency is irrelevant.

**What would change this.** Our own eval showing weak grounding, or ArmBench
adding the newer models. `pin_provider` must change with the primary model's
vendor — it is now `openai`, and was `google-ai-studio`; a vendor switch that
forgets this silently loses the determinism pin.

See [`_learnings/2026-08-01-2127_armenian-tokenizer-efficiency-inverts-llm-prices.md`](_learnings/2026-08-01-2127_armenian-tokenizer-efficiency-inverts-llm-prices.md).

---

## 11. Generation model is `google/gemini-3-flash-preview`

**Date** 2026-08-01 · **Status** **superseded by #12** (itself superseded the initial `gemini-2.5-flash` default)

**Why.** Chosen on ArmBench-LLM 1.0 *reading-comprehension* scores, not the
headline Average. Because this is a RAG system the model never recalls facts, so
the knowledge columns (Exams, MMLU, History, Literature) are irrelevant — and
those columns are most of what the Average is built from. On reading,
`gemini-2.5-flash` scored 0.4895 with Hartak at 0.4444, the worst of any capable
model on the sub-task closest to ours.

**Alternatives rejected.**
- `google/gemini-2.5-flash` — the earlier default; weakest reading of the top ten.
- `google/gemini-3-pro-preview` — best-reading Gemini (0.910) but **no longer
  sold on OpenRouter**.
- `google/gemini-3.1-pro-preview` — 0.839 vs Flash's 0.827, four times the price.
- `openai/gpt-5.4-mini` — **genuinely better** for this task (reading 0.965 vs
  0.827) and cheaper. Rejected only because the user chose Gemini. This is a
  preference, not a benchmark verdict, and the config says so.

**What would change this.** Our own eval showing weak grounding or comprehension.
`gpt-5.4-mini` is the first challenger to try; model is one line in `config.toml`
and every call is cost-logged, so the experiment is cheap. Also revisit when
ArmBench covers the newer Gemini models — `3.5-flash`, `3.6-flash` and our own
fallback `3.1-flash-lite` are currently **unmeasured on Armenian**.

See [`_knowledge/03_armenian_llm_benchmarks.md`](_knowledge/03_armenian_llm_benchmarks.md).

---

## 10. Cost comes from OpenRouter's `usage.cost`, never a local price table

**Date** 2026-08-01 · **Status** active

**Why.** Sending `usage: {"include": true}` makes OpenRouter return the
authoritative cost per call. It stays correct when prices change and when a
*fallback* model actually served the request — both of which silently corrupt a
hardcoded table.

**Alternatives rejected.** A local `{model: price}` map, which drifts invisibly;
estimating from token counts, which cannot know which model served the call.

**What would change this.** Moving off OpenRouter to a direct provider SDK that
does not return cost.

---

## 9. One OpenRouter upstream is pinned via `provider.order`

**Date** 2026-08-01 · **Status** active

**Why.** OpenRouter load-balances a single model id across backends (Google
Vertex vs AI Studio) whose greedy decoding diverges. Verified in the Washington
project: 8 identical temperature-0 calls produced 2 distinct outputs, split by
backend. Without pinning, "temperature = 0" is not reproducible and eval runs
cannot be compared.

**Alternatives rejected.** `only=[...]` with `allow_fallbacks=False` — a hard pin
that would disable failover during a real outage. We use the preference form.

**What would change this.** Evidence that the backends have converged, or a
switch away from OpenRouter.

---

## 8. Every generation knob is pinned in `config.toml`, including the prompt

**Date** 2026-08-01 · **Status** active

**Why.** The system prompt is a hyperparameter: changing it changes answers as
surely as changing temperature does, so it must be versioned and diffable
alongside the model. Nothing in `src/` hardcodes a model, threshold or prompt.
`src/config.py` reads strictly — a missing key raises at import rather than
defaulting — so a typo fails loudly.

**Alternatives rejected.** Prompt in a Python constant (not diffable as
configuration, invites ad-hoc edits); everything in `.env` (no structure, no
types, and `.env` is not committed so the values would not be reproducible).

**What would change this.** Needing per-request overrides at runtime, which would
call for layering request params over the file rather than replacing it.

---

## 7. Chunking is heading-aware and token-budgeted; nothing is ever truncated

**Date** 2026-08-01 · **Status** active

**Why.** Measured with ATE-2's real tokenizer, **90 of 94 articles (96%) exceed
the 512-token cap**, median 3,001 tokens. The archived prototype embedded whole
articles with `truncation=True`, so almost the entire corpus was silently missing
from its own vectors while the full text still went to the LLM. Chunks split on
the `## ` headings the fetcher preserves from `<h2>`, pack paragraphs to budget,
then split on sentence boundaries; the one unavoidable case (a single sentence
over budget) is logged, not hidden. Verified: 969 chunks, none over budget, and
4,318 of 4,318 source paragraphs preserved.

**Alternatives rejected.** Fixed-size windows (ignores the real section structure
these articles have); truncation (the prototype's bug); one chunk per article
(impossible — the median article is 5.9x the cap).

**What would change this.** An embedding model with a longer context window, or
eval evidence that heading boundaries hurt retrieval versus fixed windows.

---

## 6. Embeddings are ATE-2; **large** strongly indicated, pending formal eval

**Date** 2026-08-01 · **Status** measured, awaiting the eval set to confirm

**Why ATE-2.** Armenian-specific, MIT, same lab as ATE-1. Base and large share
one `tokenizer.json` (verified by sha256), so chunk boundaries are identical and
`data/chunks.jsonl` survives a switch — only the vectors and the store dimension
change (768 vs 1024).

**Evidence for large**, measured 2026-08-01 on the real corpus:

| | base (768d) | large (1024d) |
|---|---|---|
| discrimination margin, 12 Armenian pairs | +0.116 | **+0.226** |
| unrelated-topic similarity | 0.644 (too high) | 0.425 |
| relevant chunks in top-5, real query | 2 | **5** |
| score spread on a real query | 0.294 | **0.420** |
| CPU index speed | 1.1 chunks/s | 0.4 chunks/s |
| GPU (T4) index, 969 chunks | seconds | seconds |

On a real question about road-accident data, **large returned all five top hits
from the correct article; base pulled in airport queues, EV tax and passports
from rank 3**. Base's 0.644 similarity between unrelated Armenian topics means a
compressed space — and no downstream reranker can recover a chunk that never
made top-k.

**Cost of large is one-time, not per-query.** 3.3x slower to index, but a query
embeds in ~0.4s vs ~0.2s, both irrelevant interactively. On a T4 the whole index
build is seconds either way.

**Both indexes are built and kept** (`data/vectors_base.npz`,
`data/vectors_large.npz`) so the eval can measure recall@k rather than infer from
spot checks. That is what would formally settle this.

**Also settled by measuring:** `max_distance = 0.75` in config.toml filters
nothing. Unrelated content sits at cosine distance 0.356 (base) / 0.575 (large).
Realistic cutoffs are ~0.30 and ~0.45. Do not trust that config value.

---

## 5. No vector store chosen yet; hybrid retrieval is the design default

**Date** 2026-08-01 · **Status** **open — not yet decided**

**Why hybrid.** BM25 plus dense embeddings consistently beats either alone, and
that matters here specifically: Armenian proper nouns and institution names are
strong lexical signals, and the site's `infotag` taxonomy gives 1,074
hand-curated entity terms to exploit.

**What would decide it.** Whether we need BM25 in the same engine, metadata
filtering on date and `infotag`, and how much operational weight is acceptable.
The prototype used Weaviate in Docker; that is not automatically the answer.

---

## 4. Text extraction uses selectolax, not trafilatura or regex

**Date** 2026-08-01 · **Status** active

**Why.** Measured on 12 real articles rather than taken from benchmark
reputation. Public benchmarks score extractors on *whole web pages*, where the
work is finding the article among nav and footer; we pull `content.rendered` from
the REST API, which is already the body, so the real job is stripping Elementor
widgets from inside it. All three recovered the same text; selectolax is ~12x
faster than trafilatura and preserves `<h2>` boundaries as `## ` markers, which
the chunker depends on.

**Alternatives rejected.** trafilatura (flattens headings, inlines the byline
into paragraphs, 12x slower on this input); regex tag-stripping (the prototype's
approach — leaves page-builder text inline).

**What would change this.** infocom.am replacing Elementor, or a need for
full-page extraction if we ever crawl HTML instead of using the API.

---

## 3. The corpus is `indepth` (category 51), not `news`

**Date** 2026-08-01 · **Status** active

**Why.** Measured per-category: `indepth` is 5,836 long-form articles, median
1,016 words, with real journalist bylines. `news` is 146,936 posts at a median of
**72 words** under a generic `adminfo_com` account. Indexing news would fill
top-k with near-identical wire stubs; vector search has no notion of "this
document is low-information".

**Alternatives rejected.** The whole 204,581-post archive (72% wire copy, and
~160 MB / many hours of CPU embedding); `uncategorized-hy` (59,751 posts, real
articles mixed with site announcements — needs a classifier first); `reels`
(30/30 sampled were empty, video only); `infocart` (median 49 words, information
is in the image).

**What would change this.** Wanting recency coverage — a rolling window of recent
`news` alongside indepth is the obvious extension.

---

## 2. The prototype was archived, not refactored

**Date** 2026-08-01 · **Status** active

**Why.** It was built for Telegram chat messages and later pointed at a news
corpus without redesigning anything below the scraper: schema
(`TelegramMessages`, `sender`, `reply_to_message_id`), all three chunking
strategies, and the system prompt were chat-shaped. Its RAG half had never run
(no HF cache, Docker down), and its 71 passing tests mocked
`torch`/`transformers`/`weaviate` at `sys.modules` level, so they verified
nothing.

**Alternatives rejected.** Incremental refactor — the data model itself was
wrong, so the refactor would have been a rewrite with extra steps.

**Kept.** `attic/` holds it intact (`git mv`, history preserved) with
`ATTIC_NOTES.md` listing what to salvage — the WP API endpoints and the e5
prefix convention — and what never to reintroduce.

---

## 1. Notes are split across four folders by purpose

**Date** 2026-08-01 · **Status** active

**Why.** Each answers a different question and blurring them makes all of them
unsearchable.

- `_knowledge/` — researched background on a topic. "What is true about X."
- `_learnings/` — one dated gotcha per file. "What bit us, and why."
- `_work_sessions/` — one TOML per session. "What happened when."
- `DECISIONS.md` (this file) — "Why is it built this way."

**What would change this.** If `DECISIONS.md` grows past comfortable end-to-end
reading, split it into `_decisions/` the way `LEARNINGS.md` became `_learnings/`.
