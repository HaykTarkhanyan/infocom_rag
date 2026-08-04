"""Fetch a date-bounded window of the `news` feed (Լրահոս, category 49).

Separate from `fetch_articles.py` because the two corpora differ in kind, not
just in size -- see `_knowledge/04_news_section.md`:

  * ~20,500 posts a year against indepth's ~111, so this is date-bounded rather
    than "fetch the category", and it shards by month so a failure at hour two
    does not cost hour one.
  * 46% of items are republished from 16 other outlets, with the source in the
    title (`... | azatutyun.am`). That outlet is parsed into its own field: a RAG
    answering from this corpus is blending RFE/RL, the state agency and Russian
    state media, and a reader deserves to see which.
  * Posts are timestamped to the minute and recency is the whole point of
    indexing them, so local date and time are split into their own fields.

Text extraction is IMPORTED from fetch_articles rather than reimplemented, so
news and indepth are cleaned identically. Two extractors would drift, and the
drift would be invisible (see _learnings/2026-08-02-1214).

The raw `content.rendered` HTML is kept alongside the extracted text. It roughly
triples the file size and is the one thing that cannot be recovered without
re-downloading ~175 MB, so it is stored by default; drop it with --no-html.
Output is gzipped, which costs nothing to read and keeps a year near 45 MB
instead of 207 MB.

Usage:
    python src/fetch_news.py                      # last 365 days
    python src/fetch_news.py --after 2025-01-01 --before 2025-04-01
    python src/fetch_news.py --limit-months 1     # smoke test one month
    python src/fetch_news.py --force              # refetch months already on disk
"""

import argparse
import gzip
import json
import logging
import math
import re
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_articles import (  # noqa: E402
    HEADERS,
    PER_PAGE,
    REQUEST_DELAY,
    FetchError,
    api_get,
    clean_title,
    extract_content,
    fetch_term_lookup,
)

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("logs/fetch_news.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

NEWS_CATEGORY = 49
OUTPUT_DIR = Path("data/news")

POST_FIELDS = (
    "id,slug,link,date,date_gmt,modified_gmt,title,content,excerpt,"
    "author,authors,categories,tags,infotag,featured_media"
)

# Infocom republishes other outlets and marks it in the title:
#   "Ուկրաինական դրոնները կրկին հարվածել են ... | azatutyun.am"
# 16 distinct sources seen across a 700-post sample; see _knowledge/04.
SOURCE_SUFFIX = re.compile(r"\s*\|\s*([A-Za-z0-9][A-Za-z0-9.\-]*\.[a-z]{2,4})\s*$")


def split_source(title: str) -> tuple[str, str | None]:
    """Return (title without the source suffix, source outlet or None)."""
    match = SOURCE_SUFFIX.search(title)
    if not match:
        return title, None
    return title[: match.start()].strip(), match.group(1).lower()


def month_windows(after: date, before: date) -> list[tuple[str, str, str]]:
    """Calendar months covering [after, before), as (label, after_iso, before_iso).

    Sharding by month is what makes this resumable: each month is written whole
    or not at all, so an interrupted run resumes at a month boundary instead of
    restarting a 200-page walk.
    """
    windows = []
    cursor = after.replace(day=1)
    while cursor < before:
        nxt = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        start, end = max(cursor, after), min(nxt, before)
        windows.append((cursor.strftime("%Y-%m"),
                        f"{start.isoformat()}T00:00:00",
                        f"{end.isoformat()}T00:00:00"))
        cursor = nxt
    return windows


def build_record(post: dict, category_names: dict[int, str],
                 infotag_names: dict[int, str], keep_html: bool) -> dict:
    raw_html = post.get("content", {}).get("rendered", "")
    text, page_byline = extract_content(raw_html)
    title_full = clean_title(post.get("title", {}).get("rendered", ""))
    title, source = split_source(title_full)

    # `date` is the site's local time (Yerevan) and `date_gmt` is UTC. Recency is
    # the reason this corpus exists, so keep both plus the split-out local
    # date/time -- "what happened on the 3rd" should not need timezone reasoning
    # at query time.
    local = post.get("date") or ""
    record = {
        "post_id": post.get("id"),
        "slug": post.get("slug"),
        "url": post.get("link"),
        "title": title,
        "title_full": title_full,
        "source_outlet": source,          # None when it is infocom's own item
        "is_republished": source is not None,
        "date": local[:10],
        "time": local[11:16],
        "published": local,
        "published_gmt": post.get("date_gmt"),
        "modified_gmt": post.get("modified_gmt"),
        "text": text,
        "n_words": len(text.split()),
        "page_byline": page_byline,
        "author_id": post.get("author"),
        "authors": post.get("authors") or [],
        "categories": [category_names.get(c, str(c)) for c in post.get("categories", [])],
        "category_ids": post.get("categories", []),
        "infotags": [infotag_names.get(t, str(t)) for t in post.get("infotag", [])],
        "tags": post.get("tags", []),
        "featured_media": post.get("featured_media"),
    }
    if keep_html:
        record["content_html"] = raw_html
    return record


def fetch_window(after: str, before: str, category_names: dict[int, str],
                 infotag_names: dict[int, str], keep_html: bool) -> list[dict]:
    """Fetch every post in [after, before). Raises rather than returning partial."""
    probe = api_get("/posts", {"categories": NEWS_CATEGORY, "per_page": 1,
                               "after": after, "before": before, "_fields": "id"})
    total = int(probe.headers.get("X-WP-Total", 0))
    if total == 0:
        return []

    # Derived, never taken from the probe's X-WP-TotalPages: that header reflects
    # the probe's own per_page=1 and requesting a page past the real end returns
    # HTTP 400. See _learnings/2026-08-01-2052_wp-rest-totalpages-trap.md.
    pages = math.ceil(total / PER_PAGE)

    records, seen = [], set()
    for page in range(1, pages + 1):
        resp = api_get("/posts", {
            "categories": NEWS_CATEGORY, "per_page": PER_PAGE, "page": page,
            "after": after, "before": before,
            "orderby": "date", "order": "desc", "_fields": POST_FIELDS,
        })
        batch = resp.json()
        if not batch:
            break
        for post in batch:
            # New posts published mid-walk shift the window and can re-serve a
            # post on the next page. Dedupe by id rather than trusting the walk.
            if post.get("id") in seen:
                continue
            seen.add(post.get("id"))
            records.append(build_record(post, category_names, infotag_names, keep_html))
        logger.info("    page %d/%d: %d posts (%d kept)", page, pages, len(batch), len(records))
        time.sleep(REQUEST_DELAY)

    if len(records) < total * 0.95:
        raise FetchError(
            f"Expected ~{total} posts for {after[:10]}..{before[:10]} but kept "
            f"{len(records)}. Refusing to write a silently short month."
        )
    return records


def main() -> None:
    today = datetime.now(UTC).date()
    parser = argparse.ArgumentParser(description="Fetch the infocom.am news feed")
    parser.add_argument("--after", default=(today - timedelta(days=365)).isoformat(),
                        help="Inclusive start date YYYY-MM-DD (default: 365 days ago)")
    parser.add_argument("--before", default=(today + timedelta(days=1)).isoformat(),
                        help="Exclusive end date YYYY-MM-DD (default: tomorrow)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--no-html", action="store_true",
                        help="Drop raw content HTML (~3x smaller, unrecoverable without a refetch)")
    parser.add_argument("--force", action="store_true",
                        help="Refetch months that already exist on disk")
    parser.add_argument("--limit-months", type=int, help="Stop after N months (smoke test)")
    args = parser.parse_args()

    after = date.fromisoformat(args.after)
    before = date.fromisoformat(args.before)
    if after >= before:
        logger.error("--after (%s) must be before --before (%s)", after, before)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_html = not args.no_html

    logger.info("Resolving category and infotag names ...")
    category_names = fetch_term_lookup("categories")
    infotag_names = fetch_term_lookup("infotag")

    windows = month_windows(after, before)
    if args.limit_months:
        windows = windows[:args.limit_months]
    logger.info("News %s .. %s -> %d month(s), html=%s",
                after, before, len(windows), keep_html)

    grand_total = 0
    for label, win_after, win_before in windows:
        path = out_dir / f"{label}.jsonl.gz"
        if path.exists() and not args.force:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                existing = sum(1 for line in handle if line.strip())
            logger.info("%s: already on disk (%d posts) -- skipping", label, existing)
            grand_total += existing
            continue

        logger.info("%s: fetching %s .. %s", label, win_after[:10], win_before[:10])
        records = fetch_window(win_after, win_before, category_names,
                               infotag_names, keep_html)

        # Write to a temp file and rename, so an interrupted run never leaves a
        # half-written month that the resume logic would happily skip.
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp.replace(path)

        size_mb = path.stat().st_size / 1024 / 1024
        republished = sum(1 for r in records if r["is_republished"])
        logger.info("%s: wrote %d posts (%.1f MB, %d%% republished)",
                    label, len(records), size_mb, round(100 * republished / max(1, len(records))))
        grand_total += len(records)

    logger.info("")
    logger.info("=" * 64)
    logger.info("Done: %d posts across %d month(s) in %s",
                grand_total, len(windows), out_dir)
    total_mb = sum(f.stat().st_size for f in out_dir.glob("*.jsonl.gz")) / 1024 / 1024
    logger.info("On disk: %.1f MB gzipped", total_mb)
    logger.info("=" * 64)


if __name__ == "__main__":
    try:
        main()
    except FetchError as exc:
        logger.error("%s", exc)
        sys.exit(1)
