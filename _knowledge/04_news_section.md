# The `news` section (Լրահոս) — survey, 2026-08-04

Explored with Playwright + the WP REST API to assess indexing the last year of
daily news alongside (or instead of) `indepth`.

**This reopens [DECISIONS #3](../DECISIONS.md), which chose `indepth` and rejected
`news`.** Not a contradiction — #3 names this exact trigger: *"What would change
this. Wanting recency coverage — a rolling window of recent `news` alongside
indepth is the obvious extension."* But two of the premises #3 rested on turn out
to be wrong at the sizes that matter now. See "What changed" below.

---

## Headline numbers

| | all time | last 365 days |
|---|---|---|
| `news` (category **49**) | 146,960 | **20,543** |
| `indepth` (category 51) | 5,827 | **111** |

Recent volume: **~1,950/month, ~65-71/day.**

The second row is the finding that reframes the project. **`indepth` publishes
only ~111 long-form pieces a year.** Our 94-article corpus is therefore already
close to a full year of it, and no amount of indexing `indepth` will make the
system know what happened last week. Recency is only available through `news`.

---

## There are TWO news streams, and only one is in WordPress

| section | URL | backing store | reachable via |
|---|---|---|---|
| **Լրահոս** (feed) | `/news/` | WordPress, category 49 | REST API ✅ |
| **Մի շնչով** ("in one breath") | `/mi-shnchov/` → redirects to `/one-breath/` | **Telegram channel `@infocomm`** | not in WP ❌ |

`/one-breath/` is a `wptelegram/widget` embed of `t.me/infocomm`, paginated by
Telegram message id (`?before=69727`, so ~69.7k messages to date). These short
items are **not WordPress posts** and do not appear in the REST API at all.

Scraping them would mean `t.me/s/infocomm` (publicly readable) or the Telegram
API — a separate ingestion path. Worth noting the archived prototype was
originally built for Telegram messages (DECISIONS #2), so `attic/` may hold
reusable pieces.

**Not yet investigated:** whether Մի շնչով duplicates Լրահոս or carries distinct
content.

---

## What the content actually is

Sampled **700 posts** across seven windows spanning the last year:

- **46% are republished from other outlets**, with the source in the title:
  `... | azatutyun.am`. 54% carry no suffix and appear to be infocom's own.
- **16 distinct sources.** Top: `azatutyun.am` 10%, `armenpress.am` 9%,
  `factor.am` 8%, `news.am` 5%, `1lurer.am` 4%, then civilnet, shantnews,
  **arm.sputniknews.ru**, aravot, tert, pastinfo, hetq.
- Median length is the **same** either way — 120 words republished, 131 own.

**Source quality varies enormously across those 16** — `azatutyun.am` is RFE/RL,
`armenpress.am` is the state news agency, `arm.sputniknews.ru` is Russian state
media. A RAG answering from this corpus is answering from a *mixture* of
editorial standpoints, and today the system prompt has no notion of that. At
minimum the outlet should survive into chunk metadata and citations so a reader
can weigh it; it is already parseable straight out of the title.

---

## What changed since DECISIONS #3

**#3 claimed `news` has a median of 72 words.** Measured across the last year
(210 posts, stratified by month):

```
window        n  median  mean  <80w  >300w
2025-08-01..03  30      63    84    20      1
2025-10-01..03  30     141   202     4      4
2025-12-01..03  30     158   299     4      6
2026-02-01..03  30     110   153     7      2
2026-04-01..03  30      92   308    13      3
2026-06-01..03  30     116   185     7      3
2026-08-01..03  30      92   128    12      3
POOLED         210     111   194    67     22
   percentiles: p10=51  p50=111  p90=320  max=5252
```

**Median 111 words, not 72** — and the trend is upward: 2025-08 was 63, every
window since is 92-158. The original figure came from a 30-post sample of the
*whole* 146,960-post archive, which is dominated by much older, shorter items.
It was right about the archive and wrong about the recent window.

**#3's core objection was "near-identical wire stubs" swamping top-k.**
Measured on one full day (2026-08-03, 80 posts, ATE-2-large embeddings of
title+body, pairwise cosine):

| threshold | pairs | posts involved |
|---|---|---|
| > 0.95 | **0** | 0/80 |
| > 0.90 | 1 | 2/80 |
| > 0.85 | 1 | 2/80 |
| > 0.80 | 8 | 12/80 |

**Near-duplication is modest** — 2.5% of a day is near-identical, 15% loosely
similar. The one >0.90 pair was two versions of the same gas-outage notice.

**But a subtler hazard showed up in its place.** Several 0.83-0.84 pairs are
*templated* items differing only in the numbers:

```
0.838  "Արարատի մարզում վերականգնվել է 122 միլիոն 556 հազար դրամ..."
   ||  "Կոտայքի մարզում վերականգնվել է 52 միլիոն 798 հազար դրամ..."
```

A question like "how much was recovered?" will pull several of these at once,
and answering correctly depends on the model reading the region right. Not the
duplicate-swamping #3 feared — a precision problem, and one the existing prompt's
"quote exact figures" rule partly addresses. **Worth an eval case.**

---

## Fetch feasibility

Verified against the live API:

- **Deep pagination works.** `per_page=100`, pages 1 / 50 / 150 / 206 all return
  200. Page 250 returns HTTP 400 `rest_post_invalid_page_number` — past the end,
  exactly as [the `X-WP-TotalPages` learning](../_learnings/2026-08-01-2052_wp-rest-totalpages-trap.md)
  describes. Derive page count from `X-WP-Total / PER_PAGE`; never carry the
  probe's value across a `per_page` change.
- **~206 pages for the last year.** ~850 KB each → **≈175 MB of raw JSON**, at
  ~2.3-3.2s per page → **~10 minutes** of fetching. That is a real download;
  do it on wifi, not a phone hotspot.
- `after` / `before` ISO-8601 params work and are the clean way to bound a year.
- Posts carry **infotags on ~93%** of items, so the 1,074-term entity taxonomy
  survives into news and is available for filtering or hybrid retrieval.
- Two author ids appear (1, 11), not the single generic account #3 described.

**Gap in our tooling:** `src/fetch_articles.py` has `--categories` and `--limit`
but **no `--after` / `--before`**. It fetches newest-first, so `--limit` gets a
recent window by accident, but a date bound should be added before a year-scale
fetch.

### Downstream sizing, if the last year were indexed

| | current (`indepth`, 94 articles) | projected (`news`, 1 year) |
|---|---|---|
| documents | 94 | 20,543 |
| chunks | 969 | **~21,000** (median 111 words ≈ 220 tokens, so most fit one chunk) |
| vectors `.npz` | 3.6 MB | **~86 MB** (21k × 1024 dims × 4 B) |
| embedding time | seconds on a T4 | minutes on a T4; **~14h on this laptop's CPU** at 0.4 chunks/s |

**The 86 MB vector file does not belong in git** the way the current 3.6 MB one
does (the deploy builds from the repo — see DEPLOY.md). That needs deciding:
git-lfs, build-on-server, or object storage.

Query-time memory is fine: 86 MB of vectors on top of the ~790 MB measured on the
server leaves plenty inside the 4 GB box.

---

## UI mechanics (from Playwright)

- `/news/` is server-rendered — **no XHR on load**, so there is no JSON feed to
  crib from; the REST API is the right path.
- A **date picker** ("Ըստ ամսաթվի") with month/year dropdowns and a day grid
  filters the feed. Cosmetic for us — `after`/`before` on the API does the same
  job better.
- Numbered pagination at the foot of the feed.
- Each card shows time + date (`11:26`, `4 օգոստոս, 2026`), so posts are
  timestamped to the minute — useful for recency ranking.
- There is a custom `infocom/v1/news` endpoint (noted in
  [01_site_structure.md](01_site_structure.md)) that returns a cached headline
  feed. **Not yet examined** for whether it is cheaper than the WP REST route.

---

---

## What was actually fetched — 2026-08-04

`python src/fetch_news.py` pulled the full year. **Not yet chunked, embedded or
indexed** — this is the raw corpus only.

```
data/news/YYYY-MM.jsonl.gz     13 files, 16.5 MB gzipped
20,549 posts    2025-08-04 .. 2026-08-04    365 distinct days
0 duplicate post_ids           3,231,606 words total
```

Far smaller on disk than the 175 MB of JSON downloaded, because Armenian text
gzips hard: a month is ~1.2 MB. `data/*` is already gitignored, so none of this
is committed.

| field | coverage | |
|---|---|---|
| `text`, `title`, `date`, `time`, `url`, `published_gmt`, `n_words` | **100%** | one post has empty text |
| `infotags` | 94.3% | **616 distinct** tags present |
| `source_outlet` | 47.6% | **38 distinct** outlets |

- words: median **115**, mean 157, p90 283, max 18,497
- republished **48%**. Top: `armenpress.am` 2,214 · `azatutyun.am` 2,204 ·
  `factor.am` 1,303 · `1lurer.am` 1,244 · `news.am` 837 · `arm.sputniknews.ru` 481
- raw `content_html` retained on every record, so anything not extracted today
  can be recovered without refetching

**One gap in the calendar, and it is real:** no posts on 2025-08-31. The API
independently reports 0 for that Sunday, and our stored counts match it exactly
on both sides (15 / 0 / 39). Verified rather than assumed, because a missing day
is exactly what a month-boundary bug would look like.

**A bug this fetch exposed.** `extract_content` matched only block elements
(`p, h1-h5, li, ...`), so posts whose body is bare text inside `<div>`s extracted
to an empty string — silently, with no error. 7 of 1,236 in the first month
(0.6%), each holding 43-207 real words; projected ~116 lost across the year.
Zero of the 94 `indepth` articles were affected, which is why it survived until
news was fetched. Fixed with a logged fallback; the year came in with **1** empty
text instead of ~116. See
[`_learnings/2026-08-04-1120_extractor-tuned-on-one-corpus.md`](../_learnings/2026-08-04-1120_extractor-tuned-on-one-corpus.md).

## Open questions before committing to this

1. **Scope.** Whole last year (20.5k), a shorter rolling window (30-90 days), or
   news *alongside* indepth in one index? #3 suggested "rolling window alongside".
2. **Mixed corpora.** If news and indepth share an index, a question answerable
   by a considered long-form piece may instead be answered by a 100-word wire
   item. Might need source-type weighting or a retrieval filter.
3. **Republished content.** 46% is another outlet's reporting. Should the outlet
   appear in citations? (I would say yes.) Should Russian state media be
   included, excluded, or labelled?
4. **Мի շնչով / Telegram.** Separate ingestion path — worth it, or ignore?
5. **Where the vectors live**, given 86 MB and a git-based deploy.
6. **Eval.** The current 35-question set is entirely `indepth`. News questions are
   a different shape (recency, "what happened on date X", entity lookups) and
   would need their own cases.
