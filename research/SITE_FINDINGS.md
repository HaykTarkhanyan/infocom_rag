# infocom.am — site survey

Explored 2026-08-01 with Playwright + the WordPress REST API. Raw artifacts in
`research/raw/`, reproducible via `research/fetch_sitemaps.py`, `research/probe_api.py`,
and `research/probe_corpus.py` (logs land in `logs/`).

---

## TL;DR — what this survey decides

1. **The corpus to index is `indepth` (category 51), not `news`.** 5,836 articles, median
   1,016 words, real journalist bylines. The 146,936 `news` posts are 72-word wire items
   with no author. Indexing news would swamp retrieval with near-duplicate stubs.
2. **The old scraper's `author` field was wrong.** The site runs PublishPress Authors;
   real bylines live in the `ppma_author`/`authors` fields, not `author`. Every `news`
   post reports `adminfo_com`.
3. **`content.rendered` is full of Elementor page-builder markup.** Regex tag-stripping is
   not good enough; this needs real HTML parsing.
4. **Article URL slug ≠ post ID.** Post 1174257 lives at `/10045113/`.
5. **Embedding the indepth corpus on CPU is feasible** — roughly 35-40k chunks, on the
   order of an hour, not the overnight job the full 204k corpus would be.

---

## Access surface

### robots.txt

Sitemaps advertised: `sitemap_index.xml`, `news-sitemap.xml`, `wp-sitemap.xml`.
Disallowed: `/wp-admin/`, `/wp-login.php`, `/trackback/`, `/xmlrpc.php`, `/search`,
and query-string variants (`?s=`, `?replytocom=`, `?utm_`, `?fbclid=`).
Crawl-delay set only for AhrefsBot (5s), SemrushBot (5s), MJ12bot (10s) — none for `*`.

**The AI-crawler block list (GPTBot, ClaudeBot, CCBot, Google-Extended, Bytespider) is
present but entirely commented out.** The publisher left them opt-in and did not enable
them. Nothing in robots.txt disallows what we're doing. Be polite anyway: low concurrency,
identify the client.

### Sitemaps

| Sitemap | Status | Contents |
|---|---|---|
| `sitemap_index.xml` | **404** (serves an HTML error page) | advertised in robots.txt but does not exist |
| `news-sitemap.xml` | 200 | 114 recent article URLs — a rolling window, good for incremental polling |
| `wp-sitemap.xml` | 200 | index → 103 `wp-sitemap-posts-post-N.xml` shards, plus `page`, `reports`, `rewards`, `glossary`, taxonomy and user sitemaps |
| `sitemap.xml` | 200 | byte-identical to `wp-sitemap.xml` — same file, two paths |

103 post shards × ~2,000 URLs ≈ 206,000 — matches the API's post count.

### REST API

Base `https://infocom.am/wp-json`. Anonymous reads work, no auth, no rate limiting
observed at 3 concurrent workers.

45 namespaces registered. The relevant ones:

- **`wp/v2`** — the workhorse. Everything below comes from here.
- **`infocom/v1/news`** — custom cached headline feed. Returns `{posts_data, total_posts,
  max_num_pages, paged, cached_at, stale}` with title/permalink/thumbnail/date only.
  **No article body**, so it's useful for cheap change-detection, not ingestion.
- **`custom-api/v1/import-posts`**, `import-posts-publish` — write endpoints, presumably
  auth-gated. Not our business.
- **`wpml/v1`** (58 routes), **`publishpress-authors/v1`**, **`yoast/v1`**, `wc/v3`
  (WooCommerce), `mcp` (the site exposes a WordPress MCP endpoint).

Stack, per `meta[name=generator]`: WPML 4.9.5, Elementor 4.2.0, WP Rocket 3.23.1,
Site Kit by Google. Theme is Elementor-based ("hello-elementor").

---

## Site structure

### URL patterns

| Pattern | Meaning | Example |
|---|---|---|
| `/<digits>/` | **article** — the numeric part is the WP *slug*, not the post ID | `/10045113/` |
| `/category/<parent>/<child>/` | category archive | `/category/indepth/investigation/` |
| `/infotag/<slug-or-id>/` | infotag archive; slug is sometimes a number, sometimes percent-encoded Armenian | `/infotag/286/` |
| `/en/`, `/ru/` | WPML language roots | `/en/` |
| `/news/`, `/lrahos/`, `/infocards/`, `/about-us/`, `/my-account/` | section and static pages | |

**Slug vs ID matters.** Post `1174257` is served at `/10045113/`. The old scrape stored
`id: 1078157` alongside `url: https://infocom.am/10026416/`. To go from a URL back to a
post, query `?slug=10026416`.

### Categories (11 terms)

Counts are term counts from the API; they sum to more than 204,581 because posts hold
multiple categories (an `investigation` post also carries the parent `indepth`).

| Category | slug | id | posts | parent |
|---|---|---|---|---|
| Լուրեր | `news` | 49 | 146,936 | — |
| Uncategorized @hy | `uncategorized-hy` | 1 | 59,751 | — |
| **Հեղինակային** | **`indepth`** | **51** | **5,836** | — |
| Հետաքննություն | `investigation` | 1084 | 71 | 51 |
| Հետազոտություն | `research` | 1085 | 63 | 51 |
| Մի ռիլով | `reels` | 1087 | 50 | — |
| Ինֆոքարտ | `infocart` | 1086 | 41 | 51 |
| Տվյալահեն բովանդակություն | `data-driven-content` | 1083 | 41 | 51 |
| Հրատապ | `hratap` | 52 | 4 | — |
| #վճարովի_բովանդակություն | (paid content) | 1225 | 3 | — |

Querying `?categories=51` returns the whole indepth tree (parent + children), confirmed by
co-membership in the sample: `(51,1085)`, `(51,1084)`, `(51,1083,1085)`.

### Other content types

| Type | rest_base | count | useful? |
|---|---|---|---|
| post | `posts` | 204,581 | yes |
| media | `media` | 282,842 | later (images) |
| page | `pages` | 26 | marginal (about-us etc.) |
| rewards | `rewards` | 21 | no |
| reports | `reports` | 4 | maybe — likely long PDFs/analyses |
| glossary | `glossary` | 1 | no (empty in practice) |
| product | `product` | — | no (WooCommerce) |

### Taxonomies

`category` (11), `post_tag` (97), **`infotag` (1,074)**, `ppma_author` (31),
`reward-year`. `infotag` is the interesting one — it is an entity tag ("ԱՄՆ", "Իրան",
"Ռոբերտ Քոչարյան", "Ազգային ժողով"), effectively a hand-curated entity index over the
archive. That is a strong metadata filter and a cheap entity-linking signal for retrieval.

### Languages (WPML)

| lang | posts |
|---|---|
| hy | 204,581 |
| en | 277 |
| ru | 2 |

Effectively an Armenian-only archive. `?lang=all` returns the same count as `hy`.
Multilingual handling is a non-issue for *content*; it still matters for *queries*, since
users may ask in English or Russian. The e5-based embedding model is multilingual, so
cross-lingual retrieval should work without translating the corpus.

---

## Content quality by category

30-post sample per category. Word counts are of the cleaned body text.

| Category | median words | mean | min | max | empty (<20w) | bylines |
|---|---|---|---|---|---|---|
| news | 72 | 84 | 21 | 232 | 0 | `adminfo_com` ×30 |
| uncategorized-hy | 156 | 240 | 25 | 1,638 | 0 | real names |
| **indepth** | **1,016** | 1,138 | 0 | 4,573 | 3 | real names |
| investigation | 1,096 | 1,368 | 0 | 4,962 | 1 | real names |
| research | 1,558 | 1,769 | 266 | 5,052 | 0 | real names |
| data-driven-content | 1,754 | 2,007 | 211 | 5,328 | 0 | real names |
| infocart | 49 | 52 | 24 | 189 | 0 | Անի Ղևոնդյան ×29 |
| **reels** | **0** | 0 | 0 | 0 | **30/30** | Սուսինա Խաչատրյան |

Reading of this:

- **`news` is wire copy.** Short, single generic byline, high volume. Low value per token
  and high near-duplicate risk (the same agency story reworded). Excluding it removes 72%
  of the corpus and very little information.
- **`indepth` and its children are the real product** — matches the site's own tagline,
  "Հեղինակային վերլուծություն և նորություններ" (authored analysis and news). Long-form,
  attributed, section-structured.
- **`reels` is video with no text at all.** 30/30 empty. Must be excluded explicitly or it
  becomes 50 empty vectors.
- **`infocart`** is infographic cards — the text is a caption, the information is in the
  image. Excluded from a text RAG; a candidate for later multimodal work.
- **`uncategorized-hy` is a genuine grab bag**, not junk: real bylines, real articles,
  plus site announcements ("Ինֆոքոմը փնտրում է խմբագրի" — Infocom is hiring an editor) and
  a 1,638-word Scopus analysis. 59,751 posts that were simply never categorized. Mining
  this properly would need a classifier; out of scope for now.

---

## Data quality problems found

1. **Bogus dates.** 17 posts are dated `2000-01-01T00:00:00`. 19 posts predate 2015;
   24,346 predate 2020. The 2000-01-01 cluster is clearly a placeholder, not a real date.
   Any date filter or recency ranking must handle these.
2. **Duplicate posts.** `/19427/` and `/19427-2/` carry the same title and date — the
   `-2` suffix is WordPress's slug-collision pattern. Needs dedup on title+date or on
   content hash.
3. **Armenian text is HTML-entity-encoded** in `content.rendered` (`&#x531;&#x577;...`).
   `html.unescape()` handles it, but skipping that step yields unusable text.
4. **Elementor markup is embedded in the body.** A 10-article sample of indepth content
   contained 526 `<span>`, 474 `<div>`, 301 `elementor-element` classes, plus `<svg>`/
   `<path>` icon markup and `<time>` widgets. Naive `re.sub("<[^>]+>", "")` leaves widget
   text, icon-list labels and related-post titles inline with the article. **Use a real
   parser** (BeautifulSoup or selectolax) and drop Elementor chrome by class before
   extracting text.
5. **`excerpt` is not a summary.** Verified on 5 posts: the excerpt is a truncated prefix
   of the body (`excerpt is body-prefix: True` in all cases). Do not treat it as an
   abstractive summary or index it as one.
6. **`acf` is empty** (`[]`) and `meta` holds only `_acf_changed`/`footnotes`. No hidden
   structured content to mine.
7. **Some indepth posts have empty bodies** (3/30 at <20 words) — video or embed-only
   posts. Filter by word count at ingest.

---

## Useful structure for chunking

Indepth article HTML contains real semantic structure worth preserving:
`<h2>` ×20, `<blockquote>` ×19 with `<cite>` ×19, `<ul>`/`<li>` ×96 across the 10-article
sample. Section headings mean **heading-aware chunking is viable** for the indepth corpus
rather than blind fixed-size windows — chunk on `<h2>` boundaries, split further only when
a section exceeds the token budget.

---

## Field mapping for ingestion

From `/wp/v2/posts` (full sample in `research/raw/sample_post_full.json`). Available
top-level keys: `id, slug, date, date_gmt, modified, modified_gmt, link, title, content,
excerpt, author, authors, ppma_author, categories, tags, infotag, featured_media, meta,
acf, class_list, format, status, sticky, template, type, guid, _links`.

| Target field | Source | Notes |
|---|---|---|
| `post_id` | `id` | stable primary key; use for deterministic chunk UUIDs |
| `url` | `link` | |
| `slug` | `slug` | numeric string, the URL segment |
| `title` | `title.rendered` | needs `html.unescape` |
| `body` | `content.rendered` | needs entity decode **and** Elementor-aware HTML parsing |
| `published` | `date_gmt` | prefer GMT; validate against the 2000-01-01 placeholder |
| `modified` | `modified_gmt` | drives incremental re-ingest |
| `authors` | `authors[].display_name` | **not** `author` — that is the generic WP user |
| `categories` | `categories[]` | resolve ids via `research/raw/categories.json` |
| `infotags` | `infotag[]` | 1,074 entity terms; strong filter candidate |

Request with `_fields=` to keep responses small; `per_page=100` is the API maximum.

---

## Cost estimate for the indexing target

Indepth corpus: 5,836 articles, mean 1,138 words. Armenian runs ~2.2-2.8 tokens/word
through the XLM-R tokenizer (measured on real articles), so ~2,700 tokens per article and
~16M tokens total. At a 512-token chunk with overlap that is roughly **35-40k chunks**.

That is an order of magnitude smaller than the 204k-post corpus and well within reach of
CPU-only embedding on this laptop — plausibly under an hour, versus overnight for the full
archive. It also happens to sit right in the 2-5k-article target range if we scope to the
newest ~2 years, or covers the whole indepth archive if we don't.

*(The wall-clock figure is an estimate — the embedding model has not been benchmarked on
this machine yet. Worth measuring before committing.)*

---

## Open questions

1. Scope indepth to a date range, or take all 5,836?
2. Include `uncategorized-hy`? It holds real long-form work but needs classification to
   separate articles from site announcements.
3. Include `news` (146,936) at all, or keep the index analysis-only? A hybrid — index
   indepth for depth, plus a recent rolling window of news for currency — is possible.
4. `reports` (4 items) — worth a look, likely substantial PDFs.
5. Do we want `infotag` as a retrieval filter, a metadata facet, or both?
