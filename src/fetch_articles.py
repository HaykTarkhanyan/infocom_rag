"""Fetch infocom.am long-form articles via the WordPress REST API.

Defaults to the two data-heavy analysis categories:
    1085  Հետազոտություն        (research)
    1083  Տվյալահեն բովանդակություն (data-driven-content)

Usage:
    python src/fetch_articles.py
    python src/fetch_articles.py --categories 1085,1083,1084
    python src/fetch_articles.py --categories 51 --output data/indepth.jsonl
    python src/fetch_articles.py --limit 20

Output is JSONL, one article per line. Text is extracted from `content.rendered`
with selectolax after deleting page-builder chrome; `## ` marks section headings
so downstream chunking can split on them.

Design notes (see _knowledge/01_site_structure.md for the evidence):
  - Bylines come from `authors`/`ppma_author`, NOT `author`. The `author` field
    resolves to a generic `adminfo_com` account on most posts.
  - `content.rendered` carries Elementor markup, so regex tag-stripping is not
    sufficient.
  - Some posts carry a placeholder date of 2000-01-01; those are flagged.
  - Nothing here silently swallows a failure. Missing required fields raise.
"""

import argparse
import html as html_lib
import json
import logging
import math
import re
import sys
import time
from pathlib import Path

import requests
from selectolax.lexbor import LexborHTMLParser

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("logs/fetch_articles.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

API = "https://infocom.am/wp-json/wp/v2"
HEADERS = {"User-Agent": "infocom-rag research crawler (contact: tarkhanyan02@gmail.com)"}
PER_PAGE = 100          # WP REST API maximum
REQUEST_DELAY = 0.5     # seconds between page requests -- be a polite guest
TIMEOUT = 45

DEFAULT_CATEGORIES = "1085,1083"
DEFAULT_OUTPUT = "data/articles.jsonl"

PLACEHOLDER_DATE = "2000-01-01"

# Fields we ask the API for. Keeping this tight makes responses much smaller.
POST_FIELDS = (
    "id,slug,link,date,date_gmt,modified_gmt,title,content,"
    "categories,tags,infotag,authors,featured_media"
)

# Page-builder and theme chrome that is layout, not article text.
# `elementor-icon-list-*` is the post-info widget (time / date / byline). Real
# article bullets come from the text-editor widget and carry no such class.
CHROME_SELECTORS = [
    "script", "style", "noscript", "svg", "iframe", "form", "button",
    ".elementor-icon-list-items",
    ".elementor-icon-list-item",
    ".elementor-share-buttons",
    ".elementor-post__meta-data",
    ".elementor-author-box",
    ".elementor-widget-theme-post-featured-image",
    "[class*='share']",
    "[class*='related']",
    "[class*='breadcrumb']",
]

# Block-level elements whose text becomes one output block, in document order.
BLOCK_SELECTOR = "p, h1, h2, h3, h4, h5, li, blockquote, figcaption, td, th"
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5"}

# Boilerplate lines that survive extraction on some templates.
# "Read also ..." is an inline cross-reference: dropping it keeps another
# article's headline out of this article's chunks, where it would otherwise
# cause false retrieval matches.
DROP_LINE_PATTERNS = [
    re.compile(r"^\s*Կարդացեք նաև"),           # "Read also ..."
    re.compile(r"^\s*Կիսվել\s*$"),             # "Share"
    re.compile(r"^\s*(Facebook|Twitter|Telegram|LinkedIn)\s*$", re.IGNORECASE),
    re.compile(r"^\s*https?://\S+\s*$"),       # a line that is only a bare URL
    re.compile(r"^\s*[⇑↑▲⬆]\s*$"),  # back-to-top arrow
]


class FetchError(Exception):
    """Raised when the API cannot be read after retries."""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def api_get(path: str, params: dict, retries: int = 3) -> requests.Response:
    """GET a WP REST endpoint with retries. Raises FetchError when exhausted."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(f"{API}{path}", params=params,
                                timeout=TIMEOUT, headers=HEADERS)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10))
                logger.warning("Rate limited on %s, waiting %ds", path, wait)
                time.sleep(wait)
                continue
            last_error = FetchError(f"HTTP {resp.status_code} for {path}: {resp.text[:200]}")
            logger.warning("HTTP %d for %s (attempt %d/%d)",
                           resp.status_code, path, attempt, retries)
        except requests.RequestException as exc:
            last_error = exc
            logger.warning("Request error on %s: %s (attempt %d/%d)",
                           path, exc, attempt, retries)
        if attempt < retries:
            time.sleep(2 * attempt)
    raise FetchError(f"Failed to GET {path} after {retries} attempts") from last_error


def fetch_term_lookup(rest_base: str) -> dict[int, str]:
    """Fetch a whole taxonomy as {term_id: name}."""
    lookup: dict[int, str] = {}
    page = 1
    while True:
        resp = api_get(f"/{rest_base}", {"per_page": PER_PAGE, "page": page,
                                         "_fields": "id,name"})
        batch = resp.json()
        if not batch:
            break
        for term in batch:
            lookup[term["id"]] = term["name"]
        if len(batch) < PER_PAGE:
            break
        page += 1
        time.sleep(REQUEST_DELAY)
    logger.info("  %-12s %d terms", rest_base, len(lookup))
    return lookup


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_content(raw_html: str) -> tuple[str, str | None]:
    """Extract (article_text, page_byline) from a WP `content.rendered` fragment.

    Uses selectolax rather than a regex because the body is full of Elementor
    page-builder markup. `separator="", strip=False` is deliberate: the HTML's
    own whitespace is authoritative, and stripping each text node glues words
    across <span> boundaries or detaches Armenian punctuation (`օրենքի ՝`).

    The byline is read from the post-info widget before that widget is deleted.
    Many posts report `adminfo_com` in the API's `authors` field while the page
    itself credits a real journalist, so this recovers a name the API loses. It
    is identified by an `/author/` link rather than by position, and is None
    when the template carries no byline (common on recent posts).
    """
    tree = LexborHTMLParser(raw_html)

    page_byline = None
    for anchor in tree.css("a[href*='/author/']"):
        name = re.sub(r"\s+", " ", anchor.text(separator="", strip=False)).strip()
        if name:
            page_byline = name
            break

    for selector in CHROME_SELECTORS:
        for node in tree.css(selector):
            node.decompose()

    blocks: list[str] = []
    for node in tree.css(BLOCK_SELECTOR):
        text = re.sub(r"\s+", " ", node.text(separator="", strip=False)).strip()
        if not text:
            continue
        if any(pattern.match(text) for pattern in DROP_LINE_PATTERNS):
            continue
        if node.tag in HEADING_TAGS:
            blocks.append(f"## {text}")
        elif node.tag == "li":
            blocks.append(f"- {text}")
        else:
            blocks.append(text)

    # De-duplicate consecutive identical blocks (nested markup repeats text).
    deduped: list[str] = []
    for block in blocks:
        if not deduped or deduped[-1] != block:
            deduped.append(block)

    text = "\n\n".join(deduped).strip()

    if not text:
        # Some posts carry their whole body as bare text inside <div>s with no
        # <p> at all, so BLOCK_SELECTOR matches nothing and the post silently
        # becomes an empty string. Measured at 7/1236 (0.6%) of one month of
        # `news`; zero of the 94 `indepth` articles, which are Elementor-built
        # and always have <p>.
        #
        # Adding `div` to BLOCK_SELECTOR would be wrong: divs nest, so each
        # ancestor would re-emit its children's text and the dedup above only
        # collapses CONSECUTIVE repeats. Falling back to the whole tree's text
        # once, only when the structured pass found nothing, keeps the normal
        # path untouched.
        #
        # Logged at WARNING rather than applied quietly -- a second extraction
        # strategy firing is worth seeing in the fetch log.
        fallback = re.sub(r"\s+", " ", tree.text(separator=" ", strip=False)).strip()
        if fallback:
            logger.warning("No block elements matched; recovered %d chars of "
                           "unstructured text from a <div>-only body", len(fallback))
            return fallback, page_byline

    return text, page_byline


def clean_title(raw_title: str) -> str:
    return re.sub(r"\s+", " ", html_lib.unescape(raw_title)).strip()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def build_article(post: dict, category_names: dict[int, str],
                  infotag_names: dict[int, str], tag_names: dict[int, str]) -> dict:
    """Turn one API post into our article record. Raises on missing required fields."""
    for field in ("id", "slug", "link", "date_gmt", "title", "content"):
        if field not in post:
            raise KeyError(f"Post {post.get('id', '?')} is missing required field '{field}'")

    title = clean_title(post["title"]["rendered"])
    body, page_byline = extract_content(post["content"]["rendered"])

    # Real bylines live here. `author` is the generic WP account -- see _knowledge/01_site_structure.md.
    authors = [a["display_name"] for a in (post.get("authors") or [])
               if a.get("display_name")]

    date_gmt = post["date_gmt"]
    if date_gmt.startswith(PLACEHOLDER_DATE):
        logger.warning("Post %s has placeholder date %s (%s)", post["id"], date_gmt, post["link"])

    return {
        "post_id": post["id"],
        "slug": post["slug"],
        "url": post["link"],
        "title": title,
        "text": body,
        "word_count": len(body.split()),
        "published": date_gmt,
        "modified": post.get("modified_gmt", date_gmt),
        "date_is_placeholder": date_gmt.startswith(PLACEHOLDER_DATE),
        "authors": authors,
        "page_byline": page_byline,
        "categories": [category_names.get(c, f"unknown:{c}") for c in post.get("categories", [])],
        "category_ids": post.get("categories", []),
        "infotags": [infotag_names.get(t, f"unknown:{t}") for t in post.get("infotag", [])],
        "tags": [tag_names.get(t, f"unknown:{t}") for t in post.get("tags", [])],
    }


def fetch_articles(categories: str, limit: int | None) -> list[dict]:
    """Fetch every post in *categories*, newest first."""
    logger.info("Fetching taxonomy lookups...")
    category_names = fetch_term_lookup("categories")
    infotag_names = fetch_term_lookup("infotag")
    tag_names = fetch_term_lookup("tags")

    probe = api_get("/posts", {"categories": categories, "per_page": 1, "_fields": "id"})
    total = int(probe.headers.get("X-WP-Total", -1))
    if total < 0:
        raise FetchError("API did not return X-WP-Total; cannot determine corpus size")
    # Derive page count from PER_PAGE rather than trusting X-WP-TotalPages: that
    # header reflects the probe's per_page=1, not the page size we actually use.
    total_pages = math.ceil(total / PER_PAGE)
    logger.info("Categories %s contain %d posts (%d pages of %d)",
                categories, total, total_pages, PER_PAGE)

    articles: list[dict] = []
    seen_ids: set[int] = set()

    for page in range(1, total_pages + 1):
        resp = api_get("/posts", {
            "categories": categories, "per_page": PER_PAGE, "page": page,
            "orderby": "date", "order": "desc", "_fields": POST_FIELDS,
        })
        posts = resp.json()
        logger.info("  page %d/%d: %d posts", page, total_pages, len(posts))

        for post in posts:
            if post["id"] in seen_ids:
                logger.warning("Duplicate post id %s on page %d, skipping", post["id"], page)
                continue
            seen_ids.add(post["id"])
            articles.append(build_article(post, category_names, infotag_names, tag_names))
            if limit and len(articles) >= limit:
                logger.info("Reached --limit %d, stopping", limit)
                return articles

        if page < total_pages:
            time.sleep(REQUEST_DELAY)

    return articles


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def report(articles: list[dict]) -> None:
    """Log a summary so data problems are visible immediately, not at index time."""
    if not articles:
        logger.error("No articles fetched.")
        return

    counts = sorted(a["word_count"] for a in articles)
    empty = [a for a in articles if a["word_count"] < 20]
    placeholder = [a for a in articles if a["date_is_placeholder"]]
    no_author = [a for a in articles if not a["authors"]]

    logger.info("")
    logger.info("=" * 66)
    logger.info("Fetched %d articles", len(articles))
    logger.info("=" * 66)
    logger.info("  words: median=%d mean=%d min=%d max=%d total=%s",
                counts[len(counts) // 2], sum(counts) // len(counts),
                counts[0], counts[-1], f"{sum(counts):,}")
    logger.info("  date range: %s .. %s",
                min(a["published"] for a in articles)[:10],
                max(a["published"] for a in articles)[:10])
    logger.info("  articles under 20 words : %d", len(empty))
    logger.info("  placeholder dates       : %d", len(placeholder))
    logger.info("  missing byline          : %d", len(no_author))

    generic = [a for a in articles if a["authors"] == ["adminfo_com"]]
    recovered = [a for a in generic if a["page_byline"]]
    logger.info("  API byline is generic   : %d of %d", len(generic), len(articles))
    logger.info("    of those, page credits a real name: %d (still unattributed: %d)",
                len(recovered), len(generic) - len(recovered))

    bylines: dict[str, int] = {}
    for article in articles:
        for author in article["authors"]:
            bylines[author] = bylines.get(author, 0) + 1
    top = sorted(bylines.items(), key=lambda kv: -kv[1])[:8]
    logger.info("  top bylines: %s", ", ".join(f"{n} ({c})" for n, c in top))

    for article in empty:
        logger.warning("  EMPTY: %s %s", article["url"], article["title"][:60])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch infocom.am articles via the WordPress REST API",
    )
    parser.add_argument("--categories", default=DEFAULT_CATEGORIES,
                        help=f"Comma-separated category IDs (default: {DEFAULT_CATEGORIES} "
                             "= research + data-driven-content)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--limit", type=int, help="Stop after N articles")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    articles = fetch_articles(args.categories, args.limit)

    with output.open("w", encoding="utf-8") as handle:
        for article in articles:
            handle.write(json.dumps(article, ensure_ascii=False) + "\n")

    report(articles)
    logger.info("")
    logger.info("Wrote %d articles to %s", len(articles), output)


if __name__ == "__main__":
    try:
        main()
    except FetchError as exc:
        logger.error("Fetch failed: %s", exc)
        sys.exit(1)
