#!/usr/bin/env python3
"""Combine agent search results into articles.jsonl format.

Parses JSON arrays from agent output files and deduplicates by URL.
"""

import json
import re
import sys
import hashlib


def extract_json_arrays(text):
    """Extract JSON arrays from text that may contain other content."""
    articles = []
    # Try to find JSON arrays in the text
    # Look for patterns like [{"title":...}]
    bracket_depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '[' and bracket_depth == 0:
            start = i
            bracket_depth = 1
        elif ch == '[':
            bracket_depth += 1
        elif ch == ']':
            bracket_depth -= 1
            if bracket_depth == 0 and start is not None:
                try:
                    arr = json.loads(text[start:i+1])
                    if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                        articles.extend(arr)
                except json.JSONDecodeError:
                    pass
                start = None

    # Also try to find individual JSON objects
    for match in re.finditer(r'\{[^{}]*"title"[^{}]*"url"[^{}]*\}', text):
        try:
            obj = json.loads(match.group())
            if "title" in obj and "url" in obj:
                articles.append(obj)
        except json.JSONDecodeError:
            pass

    return articles


def normalize_article(article, idx):
    """Normalize an article to the expected JSONL format."""
    url = article.get("url", "")
    title = article.get("title", "")
    content = article.get("content", article.get("snippet", article.get("description", "")))
    date = article.get("date", article.get("published", ""))
    categories = article.get("categories", [])

    if not title and not content:
        return None

    # Generate a stable ID from URL or title
    id_source = url or title
    stable_id = int(hashlib.md5(id_source.encode()).hexdigest()[:8], 16)

    return {
        "id": article.get("id", stable_id),
        "title": title,
        "content": content,
        "excerpt": content[:200] if content else "",
        "author": article.get("author", "infocom.am"),
        "date": date,
        "modified": date,
        "url": url,
        "categories": categories if isinstance(categories, list) else [categories] if categories else [],
        "tags": article.get("tags", []),
        "infotags": article.get("infotags", []),
    }


def main():
    import glob

    task_dir = "/tmp/claude-0/-home-user-infocom-rag/d67743e8-ceeb-485c-9952-85b0518fc27e/tasks"
    output_file = "/home/user/infocom_rag/articles.jsonl"

    all_articles = []
    seen_urls = set()
    seen_titles = set()

    # Read all agent output files
    files = sorted(glob.glob(f"{task_dir}/*.output"))
    print(f"Found {len(files)} output files")

    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            articles = extract_json_arrays(text)
            if articles:
                print(f"  {filepath.split('/')[-1]}: {len(articles)} articles")
                all_articles.extend(articles)
        except Exception as e:
            print(f"  {filepath.split('/')[-1]}: ERROR {e}")

    print(f"\nTotal raw articles: {len(all_articles)}")

    # Deduplicate and normalize
    unique_articles = []
    for idx, article in enumerate(all_articles):
        normalized = normalize_article(article, idx)
        if normalized is None:
            continue

        url = normalized["url"]
        title = normalized["title"]

        # Dedup by URL first, then by title
        if url and url in seen_urls:
            continue
        if title and title in seen_titles:
            continue

        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)
        unique_articles.append(normalized)

    print(f"Unique articles after dedup: {len(unique_articles)}")

    # Write output
    with open(output_file, "w", encoding="utf-8") as f:
        for article in unique_articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    print(f"Written to {output_file}")


if __name__ == "__main__":
    main()
