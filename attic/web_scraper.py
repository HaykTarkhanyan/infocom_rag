"""Scrape all articles from infocom.am via the WordPress REST API.

Usage:
    python web_scraper.py                              # scrape all articles
    python web_scraper.py --limit 1000                 # first N articles (newest first)
    python web_scraper.py --output articles.jsonl      # custom output path
    python web_scraper.py --resume                     # resume interrupted scrape
    python web_scraper.py --workers 5                  # concurrent API requests
    python web_scraper.py --category news              # only a specific category

Output is JSONL (one JSON object per line), compatible with ingest.py:
    python ingest.py articles.jsonl
"""

import argparse
import html as html_lib
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

WP_API_BASE = "https://infocom.am/wp-json/wp/v2"
POSTS_PER_PAGE = 100  # WP REST API maximum
DEFAULT_OUTPUT = "articles.jsonl"
DEFAULT_LOG_FILE = "web_scraper.log"
DEFAULT_WORKERS = 3
RETRY_LIMIT = 3
RETRY_DELAY = 5  # seconds

logger = logging.getLogger("web_scraper")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def html_to_text(raw_html: str) -> str:
    """Convert HTML to plain text, preserving paragraph breaks."""
    text = re.sub(r"<br\s*/?>", "\n", raw_html)
    text = re.sub(r"</p>", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_lib.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class FetchError(Exception):
    """Raised when an HTTP request fails after all retries."""


def fetch_json(
    url: str,
    params: dict | None = None,
    retries: int = RETRY_LIMIT,
) -> list | dict:
    """GET JSON from *url* with retry and rate-limit handling.

    Raises FetchError if all retries are exhausted.
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", RETRY_DELAY))
                logger.warning("Rate limited, waiting %ds...", wait)
                time.sleep(wait)
                continue
            if resp.status_code == 400 and "rest_post_invalid_page_number" in resp.text:
                return []  # past the last page
            last_error = FetchError(f"HTTP {resp.status_code} for {url}")
            logger.warning("HTTP %d for %s (attempt %d/%d)", resp.status_code, url, attempt + 1, retries)
        except requests.RequestException as e:
            last_error = e
            logger.warning("Request error: %s (attempt %d/%d)", e, attempt + 1, retries)
        if attempt < retries - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))
    raise FetchError(
        f"Failed to fetch {url} after {retries} attempts"
    ) from last_error


def fetch_lookup(endpoint: str) -> dict[int, str]:
    """Fetch all items from a paginated WP taxonomy/user endpoint → {id: name}.

    Raises FetchError if any page request fails.
    Raises KeyError if an item is missing 'id' or 'name'.
    """
    lookup = {}
    page = 1
    while True:
        data = fetch_json(f"{WP_API_BASE}/{endpoint}", {"per_page": 100, "page": page})
        if not data:
            break
        for item in data:
            if "id" not in item:
                raise KeyError(f"Item from '{endpoint}' page {page} is missing 'id': {item}")
            if "name" not in item:
                raise KeyError(f"Item id={item['id']} from '{endpoint}' is missing 'name'")
            lookup[item["id"]] = item["name"]
        if len(data) < 100:
            break
        page += 1
    return lookup


# ---------------------------------------------------------------------------
# Core scraping
# ---------------------------------------------------------------------------


def get_total_posts(category_id: int | None = None) -> tuple[int, int]:
    """Return (total_posts, total_pages) via a HEAD request.

    Raises FetchError if the API doesn't return pagination headers.
    """
    params: dict = {"per_page": POSTS_PER_PAGE}
    if category_id is not None:
        params["categories"] = category_id
    resp = requests.head(f"{WP_API_BASE}/posts", params=params, timeout=30)
    resp.raise_for_status()
    if "X-WP-Total" not in resp.headers:
        raise FetchError(
            f"WordPress API did not return X-WP-Total header (status {resp.status_code})"
        )
    total = int(resp.headers["X-WP-Total"])
    pages = int(resp.headers["X-WP-TotalPages"])
    return total, pages


def fetch_page(
    page: int,
    category_id: int | None,
    author_map: dict[int, str],
    category_map: dict[int, str],
    tag_map: dict[int, str],
    infotag_map: dict[int, str],
) -> list[dict]:
    """Fetch one page of posts and return a list of article dicts."""
    params: dict = {
        "per_page": POSTS_PER_PAGE,
        "page": page,
        "_fields": "id,date,modified,title,content,excerpt,author,categories,tags,infotag,link",
    }
    if category_id is not None:
        params["categories"] = category_id

    data = fetch_json(f"{WP_API_BASE}/posts", params)
    if not data:
        logger.debug("Page %d returned no data", page)
        return []

    logger.debug("Page %d: fetched %d posts", page, len(data))
    articles = []
    for post in data:
        for field in ("id", "title", "content", "date", "author", "link"):
            if field not in post:
                raise KeyError(f"Post on page {page} is missing required field '{field}'")

        title = html_lib.unescape(post["title"]["rendered"]).strip()
        content = html_to_text(post["content"]["rendered"])
        excerpt = html_to_text(post.get("excerpt", {}).get("rendered", ""))

        if not content and not title:
            continue

        author_id = post["author"]
        if author_id not in author_map:
            raise KeyError(f"Post id={post['id']}: author id={author_id} not found in author lookup")

        cat_ids = post.get("categories", [])
        tag_ids = post.get("tags", [])
        infotag_ids = post.get("infotag", [])

        articles.append({
            "id": post["id"],
            "title": title,
            "content": content,
            "excerpt": excerpt,
            "author": author_map[author_id],
            "date": post["date"],
            "modified": post.get("modified", post["date"]),
            "url": post["link"],
            "categories": [category_map[c] for c in cat_ids],
            "tags": [tag_map[t] for t in tag_ids],
            "infotags": [infotag_map[t] for t in infotag_ids],
        })

    return articles


def load_existing_ids(path: str) -> set[int]:
    """Load article IDs already present in the output file.

    Raises KeyError if any non-empty line is missing the 'id' field.
    Raises json.JSONDecodeError if any non-empty line is not valid JSON.
    Returns an empty set if the file does not exist.
    """
    try:
        f = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        return set()

    ids = set()
    with f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj:
                raise KeyError(f"Line {line_num} in {path} is missing required 'id' field")
            ids.add(obj["id"])
    return ids


def resolve_category_id(name: str, category_map: dict[int, str]) -> int | None:
    """Look up a category by name, slug, or ID string."""
    # Try as numeric ID
    try:
        cid = int(name)
        if cid in category_map:
            return cid
    except ValueError:
        pass
    # Try exact name match (case-insensitive)
    for cid, cname in category_map.items():
        if cname.lower() == name.lower():
            return cid
    # Try slug-style match
    for cid, cname in category_map.items():
        if name.lower().replace(" ", "-") in cname.lower().replace(" ", "-"):
            return cid
    return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def scrape(
    output_path: str,
    limit: int | None,
    workers: int,
    category: str | None,
    resume: bool,
) -> None:
    """Scrape articles from infocom.am and write JSONL output."""
    logger.info("Fetching lookups (categories, authors, tags, infotags)...")
    category_map = fetch_lookup("categories")
    author_map = fetch_lookup("users")
    tag_map = fetch_lookup("tags")
    infotag_map = fetch_lookup("infotag")
    logger.info("  %d categories, %d authors, %d tags, %d infotags", len(category_map), len(author_map), len(tag_map), len(infotag_map))

    # Resolve category filter
    category_id = None
    if category:
        category_id = resolve_category_id(category, category_map)
        if category_id is None:
            logger.error("Category '%s' not found. Available: %s", category, sorted(category_map.values()))
            sys.exit(1)
        logger.info("Filtering by category: %s (id=%d)", category_map[category_id], category_id)

    total_posts, total_pages = get_total_posts(category_id)
    logger.info("Total articles: %s (%s pages of %d)", f"{total_posts:,}", f"{total_pages:,}", POSTS_PER_PAGE)

    if limit:
        max_pages = (limit + POSTS_PER_PAGE - 1) // POSTS_PER_PAGE
        total_pages = min(total_pages, max_pages)
        logger.info("Limiting to ~%d articles (%d pages)", limit, total_pages)

    # Resume: load existing article IDs to skip duplicates
    seen_ids: set[int] = set()
    article_count = 0
    if resume:
        seen_ids = load_existing_ids(output_path)
        if seen_ids:
            article_count = len(seen_ids)
            logger.info("Resume: %s articles already on disk, will skip duplicates", f"{article_count:,}")

    mode = "a" if resume and seen_ids else "w"
    pages_to_fetch = list(range(1, total_pages + 1))

    if not pages_to_fetch:
        logger.info("Nothing to fetch.")
        return

    new_count = 0
    with open(output_path, mode, encoding="utf-8") as f:
        batch_size = max(workers * 2, 4)

        for batch_start in range(0, len(pages_to_fetch), batch_size):
            batch_pages = pages_to_fetch[batch_start : batch_start + batch_size]

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        fetch_page, pg, category_id, author_map, category_map, tag_map, infotag_map
                    ): pg
                    for pg in batch_pages
                }

                # Collect results in page order for deterministic output
                page_results: dict[int, list[dict]] = {}
                for future in as_completed(futures):
                    pg = futures[future]
                    page_results[pg] = future.result()

                for pg in sorted(page_results):
                    for article in page_results[pg]:
                        if limit and (article_count + new_count) >= limit:
                            break
                        if article["id"] in seen_ids:
                            continue
                        seen_ids.add(article["id"])
                        f.write(json.dumps(article, ensure_ascii=False) + "\n")
                        new_count += 1
                    if limit and (article_count + new_count) >= limit:
                        break

            f.flush()
            last_page = batch_pages[-1] if batch_pages else 0
            total_on_disk = article_count + new_count
            logger.info("  %s articles on disk (%s new)  (page %d/%d)", f"{total_on_disk:,}", f"{new_count:,}", last_page, total_pages)

            if limit and total_on_disk >= limit:
                break

    logger.info("Done. %s new articles written to %s (%s total)", f"{new_count:,}", output_path, f"{article_count + new_count:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape articles from infocom.am via WordPress REST API"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--limit", type=int,
        help="Max number of articles to scrape",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Concurrent API requests (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--category",
        help="Filter by category name, slug, or ID (e.g. 'news', 'indepth')",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from where a previous run left off",
    )
    parser.add_argument(
        "--log-file", default=DEFAULT_LOG_FILE,
        help=f"Log file path (default: {DEFAULT_LOG_FILE})",
    )
    args = parser.parse_args()

    # Set up logging: file (detailed) + console (progress)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(args.log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    scrape(args.output, args.limit, args.workers, args.category, args.resume)


if __name__ == "__main__":
    main()
