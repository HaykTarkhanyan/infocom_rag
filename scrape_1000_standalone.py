#!/usr/bin/env python3
"""Standalone fast scraper for 1000 articles from infocom.am.

This script bypasses any system proxy and uses aggressive parallelism.
Run on a machine with direct internet access:

    python3 scrape_1000_standalone.py

Output: articles.jsonl (JSONL format, compatible with ingest.py)
"""

import html as html_lib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Bypass any proxy settings
for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(var, None)

import requests

WP_API_BASE = "https://infocom.am/wp-json/wp/v2"
POSTS_PER_PAGE = 100
TARGET_ARTICLES = 1000
WORKERS = 8
OUTPUT = "articles.jsonl"
RETRY_LIMIT = 3
RETRY_DELAY = 3


def html_to_text(raw_html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", raw_html)
    text = re.sub(r"</p>", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_lib.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_json(url, params=None):
    for attempt in range(RETRY_LIMIT):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", RETRY_DELAY))
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 400 and "rest_post_invalid_page_number" in resp.text:
                return []
        except requests.RequestException as e:
            print(f"  Request error (attempt {attempt+1}/{RETRY_LIMIT}): {e}")
        if attempt < RETRY_LIMIT - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {RETRY_LIMIT} attempts")


def fetch_lookup(endpoint):
    lookup = {}
    page = 1
    while True:
        data = fetch_json(f"{WP_API_BASE}/{endpoint}", {"per_page": 100, "page": page})
        if not data:
            break
        for item in data:
            lookup[item["id"]] = item["name"]
        if len(data) < 100:
            break
        page += 1
    return lookup


def fetch_page(page, category_id, author_map, category_map, tag_map, infotag_map):
    params = {
        "per_page": POSTS_PER_PAGE,
        "page": page,
        "_fields": "id,date,modified,title,content,excerpt,author,categories,tags,infotag,link",
    }
    if category_id is not None:
        params["categories"] = category_id

    data = fetch_json(f"{WP_API_BASE}/posts", params)
    if not data:
        return []

    articles = []
    for post in data:
        title = html_lib.unescape(post["title"]["rendered"]).strip()
        content = html_to_text(post["content"]["rendered"])
        excerpt = html_to_text(post.get("excerpt", {}).get("rendered", ""))

        if not content and not title:
            continue

        author_id = post["author"]
        articles.append({
            "id": post["id"],
            "title": title,
            "content": content,
            "excerpt": excerpt,
            "author": author_map.get(author_id, f"author_{author_id}"),
            "date": post["date"],
            "modified": post.get("modified", post["date"]),
            "url": post["link"],
            "categories": [category_map.get(c, str(c)) for c in post.get("categories", [])],
            "tags": [tag_map.get(t, str(t)) for t in post.get("tags", [])],
            "infotags": [infotag_map.get(t, str(t)) for t in post.get("infotag", [])],
        })
    return articles


def main():
    print(f"=== Scraping {TARGET_ARTICLES} articles from infocom.am ===")
    print(f"Workers: {WORKERS} | Output: {OUTPUT}")
    print()

    # Load existing articles for resume
    seen_ids = set()
    if os.path.exists(OUTPUT):
        with open(OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seen_ids.add(json.loads(line)["id"])
        if seen_ids:
            print(f"Resume: {len(seen_ids)} articles already on disk")
            if len(seen_ids) >= TARGET_ARTICLES:
                print("Already have enough articles!")
                return

    print("Fetching lookups...")
    category_map = fetch_lookup("categories")
    author_map = fetch_lookup("users")
    tag_map = fetch_lookup("tags")
    infotag_map = fetch_lookup("infotag")
    print(f"  {len(category_map)} categories, {len(author_map)} authors, {len(tag_map)} tags, {len(infotag_map)} infotags")

    # Get total posts
    resp = requests.head(f"{WP_API_BASE}/posts", params={"per_page": POSTS_PER_PAGE}, timeout=30)
    resp.raise_for_status()
    total_posts = int(resp.headers["X-WP-Total"])
    total_pages = int(resp.headers["X-WP-TotalPages"])
    max_pages = min(total_pages, (TARGET_ARTICLES + POSTS_PER_PAGE - 1) // POSTS_PER_PAGE)
    print(f"  {total_posts} total articles, fetching up to {max_pages} pages")
    print()

    new_count = 0
    mode = "a" if seen_ids else "w"
    pages_to_fetch = list(range(1, max_pages + 1))

    with open(OUTPUT, mode, encoding="utf-8") as f:
        batch_size = WORKERS * 2

        for batch_start in range(0, len(pages_to_fetch), batch_size):
            batch_pages = pages_to_fetch[batch_start:batch_start + batch_size]

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = {
                    pool.submit(fetch_page, pg, None, author_map, category_map, tag_map, infotag_map): pg
                    for pg in batch_pages
                }

                page_results = {}
                for future in as_completed(futures):
                    pg = futures[future]
                    try:
                        page_results[pg] = future.result()
                    except Exception as e:
                        print(f"  ERROR on page {pg}: {e}")
                        page_results[pg] = []

                for pg in sorted(page_results):
                    for article in page_results[pg]:
                        if (len(seen_ids) + new_count) >= TARGET_ARTICLES:
                            break
                        if article["id"] in seen_ids:
                            continue
                        seen_ids.add(article["id"])
                        f.write(json.dumps(article, ensure_ascii=False) + "\n")
                        new_count += 1

            f.flush()
            total = len(seen_ids)
            last_pg = batch_pages[-1]
            print(f"  {total} articles ({new_count} new) — page {last_pg}/{max_pages}")

            if total >= TARGET_ARTICLES:
                break

    print(f"\nDone! {new_count} new articles written to {OUTPUT} ({len(seen_ids)} total)")


if __name__ == "__main__":
    main()
