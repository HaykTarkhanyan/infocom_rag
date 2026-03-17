#!/usr/bin/env python3
"""Parse agent output transcripts to extract infocom.am article data.

Extracts URLs, titles, dates, and content snippets from WebSearch results
embedded in agent progress logs.
"""

import glob
import hashlib
import json
import re
import os


def extract_search_results(text):
    """Extract article data from agent transcript text."""
    articles = []

    # Pattern 1: Extract from search result links JSON
    # Look for "links" arrays in search results
    for match in re.finditer(r'"links":\s*\[(.*?)\]', text, re.DOTALL):
        try:
            links_str = "[" + match.group(1) + "]"
            # Clean up escaped characters
            links_str = links_str.replace('\\"', '"').replace('\\n', '\n')
            links = json.loads(links_str)
            for link in links:
                if isinstance(link, dict) and "url" in link:
                    url = link["url"]
                    if "infocom.am" in url:
                        articles.append({
                            "title": link.get("title", ""),
                            "url": url,
                        })
        except (json.JSONDecodeError, KeyError):
            pass

    # Pattern 2: Extract URLs with titles from markdown-style links
    for match in re.finditer(r'\[([^\]]+)\]\((https?://[^\s)]*infocom\.am[^\s)]*)\)', text):
        title = match.group(1).strip()
        url = match.group(2).strip().rstrip('\\')
        if title and url:
            articles.append({"title": title, "url": url})

    # Pattern 3: Extract from "title":"..." "url":"..." patterns
    for match in re.finditer(r'"title"\s*:\s*"([^"]+)"[^}]*"url"\s*:\s*"([^"]*infocom\.am[^"]*)"', text):
        articles.append({"title": match.group(1), "url": match.group(2)})

    # Also reverse order
    for match in re.finditer(r'"url"\s*:\s*"([^"]*infocom\.am[^"]*)"[^}]*"title"\s*:\s*"([^"]+)"', text):
        articles.append({"title": match.group(2), "url": match.group(1)})

    # Pattern 4: Extract content snippets near URLs
    # Look for descriptions/summaries near infocom.am URLs
    content_map = {}
    for match in re.finditer(
        r'(https?://(?:new\.)?infocom\.am/(?:en/[Aa]rticle/)?(\d+)/?)["\s\\]*[^"]{0,50}?(?:content|description|snippet|text)["\s:]*"([^"]{20,500})"',
        text
    ):
        url = match.group(1).rstrip('\\/')
        content = match.group(3)
        if url not in content_map or len(content) > len(content_map[url]):
            content_map[url] = content

    # Pattern 5: Extract longer text blocks that describe articles
    for match in re.finditer(
        r'(?:coverage of|article about|report[s]? on|discussing|addresses)\s+(.{30,300}?)(?:\.|\\n|")',
        text, re.IGNORECASE
    ):
        pass  # Just for additional context

    return articles, content_map


def extract_descriptions_from_search_text(text):
    """Extract article descriptions from the AI-processed search result text."""
    descriptions = {}

    # Find blocks that describe articles with URLs
    # Pattern: **[Title](URL)** — Description
    for match in re.finditer(
        r'\*\*\[([^\]]+)\]\((https?://[^\s)]*infocom\.am[^\s)]*)\)\*\*\s*[-–—]\s*([^\n]+)',
        text
    ):
        title = match.group(1)
        url = match.group(2).rstrip('\\')
        desc = match.group(3).strip()
        descriptions[url] = {"title": title, "content": desc}

    # Pattern: N. **[Title](URL)** - Description
    for match in re.finditer(
        r'\d+\.\s*\*?\*?\[([^\]]+)\]\((https?://[^\s)]*infocom\.am[^\s)]*)\)\*?\*?\s*[-–—:]\s*([^\n]+)',
        text
    ):
        title = match.group(1)
        url = match.group(2).rstrip('\\')
        desc = match.group(3).strip()
        if url not in descriptions:
            descriptions[url] = {"title": title, "content": desc}

    return descriptions


def clean_url(url):
    """Clean up URL artifacts."""
    url = url.rstrip('\\/)')
    url = re.sub(r'\\[nt].*', '', url)
    url = re.sub(r'\*+$', '', url)
    return url.strip()


def url_to_id(url):
    """Extract numeric ID from URL or generate one."""
    match = re.search(r'/(\d+)/?$', url)
    if match:
        return int(match.group(1))
    match = re.search(r'/[Aa]rticle/(\d+)', url)
    if match:
        return int(match.group(1))
    return int(hashlib.md5(url.encode()).hexdigest()[:8], 16)


def main():
    task_dir = "/tmp/claude-0/-home-user-infocom-rag/d67743e8-ceeb-485c-9952-85b0518fc27e/tasks"
    output_file = "/home/user/infocom_rag/articles.jsonl"

    all_articles = {}  # url -> article dict
    all_content = {}  # url -> content string

    files = sorted(glob.glob(f"{task_dir}/a*.output"))
    print(f"Parsing {len(files)} agent output files...")

    for filepath in files:
        fname = os.path.basename(filepath)
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            articles, content_map = extract_search_results(text)
            descriptions = extract_descriptions_from_search_text(text)

            # Merge content
            for url, content in content_map.items():
                clean = clean_url(url)
                if clean not in all_content or len(content) > len(all_content.get(clean, "")):
                    all_content[clean] = content

            for url, desc in descriptions.items():
                clean = clean_url(url)
                if clean not in all_content or len(desc.get("content", "")) > len(all_content.get(clean, "")):
                    all_content[clean] = desc.get("content", "")

            found = 0
            for article in articles:
                url = clean_url(article["url"])
                title = article.get("title", "").strip()

                # Skip non-article pages
                if not url or "/infotag/" in url or "/Home/" in url:
                    continue
                if url in ("https://infocom.am", "https://infocom.am/",
                           "https://infocom.am/en", "https://infocom.am/en/",
                           "https://new.infocom.am"):
                    continue

                if url not in all_articles:
                    all_articles[url] = {
                        "title": title,
                        "url": url,
                    }
                    found += 1
                elif title and not all_articles[url].get("title"):
                    all_articles[url]["title"] = title

            print(f"  {fname}: {found} new articles (total: {len(all_articles)})")
        except Exception as e:
            print(f"  {fname}: ERROR {e}")

    # Build final JSONL
    print(f"\nTotal unique article URLs: {len(all_articles)}")
    print(f"Articles with content: {len(all_content)}")

    output_articles = []
    for url, article in sorted(all_articles.items()):
        title = article.get("title", "")
        content = all_content.get(url, "")

        # Clean title
        title = title.replace(" - Infocom", "").replace(" - infocom.am", "").strip()
        title = re.sub(r'\\[nt]', '', title)

        article_id = url_to_id(url)

        output_articles.append({
            "id": article_id,
            "title": title,
            "content": content,
            "excerpt": content[:200] if content else "",
            "author": "infocom.am",
            "date": "",
            "modified": "",
            "url": url,
            "categories": [],
            "tags": [],
            "infotags": [],
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for article in output_articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(output_articles)} articles to {output_file}")


if __name__ == "__main__":
    main()
