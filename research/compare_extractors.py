"""Compare HTML-to-text extractors on real infocom.am article bodies.

Public benchmarks (trafilatura's, scrapinghub's) score extractors on *whole web
pages*, where most of the work is finding the article among nav/header/footer.
We pull `content.rendered` from the WP REST API, which is already just the body --
the remaining job is stripping Elementor page-builder widgets from *inside* it.
That is a different task, so this measures the candidates on our actual input.

Candidates:
  regex       -- what the archived prototype did (attic/web_scraper.py)
  selectolax  -- DOM parse, drop chrome by tag/class, then extract text
  trafilatura -- the benchmark leader for full-page article extraction

Usage:
    python research/compare_extractors.py
"""

import html as html_lib
import json
import logging
import re
import time
from pathlib import Path

import requests
import trafilatura
from selectolax.lexbor import LexborHTMLParser

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("logs/compare_extractors.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

API = "https://infocom.am/wp-json/wp/v2/posts"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; infocom_rag)"}
RAW = Path("research/raw")

# Categories: 1085 = research (Հետազոտություն), 1083 = data-driven-content
SAMPLE_CATEGORIES = "1085,1083"
SAMPLE_SIZE = 12

# Elementor / theme chrome that is structure, not article text.
CHROME_SELECTORS = [
    "script", "style", "noscript", "svg", "iframe", "form", "button",
    ".elementor-widget-heading + .elementor-widget-post-info",
    ".elementor-post__meta-data",
    ".elementor-share-buttons",
    ".elementor-icon-list-items",
    ".elementor-widget-theme-post-featured-image",
    ".elementor-author-box",
    "[class*='share']",
    "[class*='related']",
    "[class*='breadcrumb']",
]

# Strings that should NOT survive extraction -- if they do, boilerplate leaked through.
BOILERPLATE_MARKERS = [
    "Կարդացեք նաև",     # "Read also"
    "Կիսվել",            # "Share"
    "Facebook", "Twitter", "Telegram",
    "elementor",
]


def extract_regex(raw_html: str) -> str:
    """The archived prototype's approach: strip tags with a regex."""
    text = re.sub(r"<br\s*/?>", "\n", raw_html)
    text = re.sub(r"</p>", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_lib.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_selectolax(raw_html: str) -> str:
    """Parse the DOM, delete chrome, then read text with block structure kept."""
    tree = LexborHTMLParser(raw_html)
    for selector in CHROME_SELECTORS:
        for node in tree.css(selector):
            node.decompose()

    blocks = []
    for node in tree.css("p, h1, h2, h3, h4, li, blockquote, figcaption"):
        text = node.text(separator=" ", strip=True)
        if not text:
            continue
        if node.tag in ("h1", "h2", "h3", "h4"):
            blocks.append(f"\n## {text}")
        elif node.tag == "li":
            blocks.append(f"- {text}")
        else:
            blocks.append(text)

    if not blocks:  # no block elements at all -- fall back to whole-tree text
        body = tree.body
        return body.text(separator="\n", strip=True) if body else ""

    out = "\n\n".join(blocks)
    out = re.sub(r"[ \t]+", " ", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def extract_trafilatura(raw_html: str) -> str:
    """Trafilatura's article extractor.

    Returns "" when trafilatura declines to extract, which it does on short or
    fragment-like input. That outcome is reported, not hidden.
    """
    result = trafilatura.extract(
        raw_html,
        include_comments=False,
        include_tables=True,
        include_formatting=False,
        favor_recall=True,
    )
    return result or ""


EXTRACTORS = {
    "regex": extract_regex,
    "selectolax": extract_selectolax,
    "trafilatura": extract_trafilatura,
}


def fetch_sample() -> list[dict]:
    resp = requests.get(API, params={
        "categories": SAMPLE_CATEGORIES,
        "per_page": SAMPLE_SIZE,
        "_fields": "id,link,title,content",
    }, timeout=45, headers=HEADERS)
    resp.raise_for_status()
    posts = resp.json()
    logger.info("Fetched %d articles from categories %s", len(posts), SAMPLE_CATEGORIES)
    return posts


def main():
    posts = fetch_sample()

    stats = {name: {"words": [], "secs": 0.0, "empty": 0, "leaks": 0} for name in EXTRACTORS}
    per_article = []

    for post in posts:
        raw = post["content"]["rendered"]
        row = {"id": post["id"], "link": post["link"], "raw_chars": len(raw)}

        for name, func in EXTRACTORS.items():
            start = time.perf_counter()
            text = func(raw)
            elapsed = time.perf_counter() - start

            words = len(text.split())
            leaked = [m for m in BOILERPLATE_MARKERS if m in text]

            stats[name]["words"].append(words)
            stats[name]["secs"] += elapsed
            if words == 0:
                stats[name]["empty"] += 1
            if leaked:
                stats[name]["leaks"] += 1

            row[name] = {"words": words, "leaked": leaked, "text": text}

        per_article.append(row)

    logger.info("")
    logger.info("=" * 74)
    logger.info("Results over %d articles", len(posts))
    logger.info("=" * 74)
    logger.info("%-12s %8s %8s %8s %8s %8s", "extractor", "medianW", "totalW", "empty", "leaky", "secs")
    for name, s in stats.items():
        words = sorted(s["words"])
        median = words[len(words) // 2] if words else 0
        logger.info("%-12s %8d %8d %8d %8d %8.3f",
                    name, median, sum(s["words"]), s["empty"], s["leaks"], s["secs"])

    logger.info("")
    logger.info("Per-article word counts (regex / selectolax / trafilatura):")
    for row in per_article:
        logger.info("  %-9s raw=%-7s %6d / %6d / %6d   %s",
                    row["id"], row["raw_chars"],
                    row["regex"]["words"], row["selectolax"]["words"],
                    row["trafilatura"]["words"], row["link"])

    logger.info("")
    logger.info("Boilerplate leaks by extractor:")
    for name in EXTRACTORS:
        leaks = {}
        for row in per_article:
            for marker in row[name]["leaked"]:
                leaks[marker] = leaks.get(marker, 0) + 1
        logger.info("  %-12s %s", name, leaks or "none")

    out = RAW / "extractor_comparison.json"
    out.write_text(json.dumps(per_article, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("")
    logger.info("Full text of every extraction saved to %s", out)


if __name__ == "__main__":
    main()
