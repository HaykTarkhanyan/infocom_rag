# Design decisions

Newest first. Deep supporting research lives in [`_knowledge/`](_knowledge/README.md);
this file holds the choice and a pointer.

Entries 1-11 were written on 2026-08-01 covering decisions made earlier the same
day, before this file existed. Later entries are recorded as the decision is made.

---

## 11. Generation model is `google/gemini-3-flash-preview`

**Date** 2026-08-01 · **Status** active (supersedes the initial `gemini-2.5-flash` default)

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

## 6. Embeddings are ATE-2; base vs large is still undecided

**Date** 2026-08-01 · **Status** **open — not yet decided**

**Why.** `Metric-AI/armenian-text-embeddings-2` is Armenian-specific, MIT, and
the same lab's ATE-1 was already in use. Base and large share one `tokenizer.json`
(verified by sha256), so chunk boundaries are identical between them and
`data/chunks.jsonl` survives a switch — only the vectors and the store's
dimension change (768 vs 1024).

**Open because** weights are not downloaded (bandwidth) and neither has been
benchmarked on this machine.

**What would decide it.** A speed benchmark on this CPU plus retrieval quality on
our eval set. Large is ~2x slower on CPU; if quality is a wash, base wins.

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
