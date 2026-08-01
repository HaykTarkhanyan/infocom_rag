"""Second-pass probe: which slice of infocom.am is worth indexing?

The first probe found 204,581 posts split very unevenly across categories.
This one characterises the candidate slices so we can pick a 2-5k subset:
  - Հեղինակային / indepth (5,836) -- original authored analysis
  - Լուրեր / news (146,936)       -- wire-style news
  - Uncategorized @hy (59,751)    -- what IS this bucket?
and checks data quality: bogus dates, empty bodies, real bylines, dupes.
"""

import html as html_lib
import json
import logging
import re
import statistics
from collections import Counter
from pathlib import Path

import requests

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("logs/probe_corpus.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

API = "https://infocom.am/wp-json"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; infocom_rag)"}
RAW = Path("research/raw")
RAW.mkdir(parents=True, exist_ok=True)

CATEGORIES = {
    "news (Լուրեր)": 49,
    "uncategorized-hy": 1,
    "indepth (Հեղինակային)": 51,
    "investigation": 1084,
    "research": 1085,
    "infocart": 1086,
    "data-driven": 1083,
    "reels": 1087,
}


def get(path: str, params: dict | None = None) -> requests.Response:
    resp = requests.get(f"{API}{path}", params=params, timeout=45, headers=HEADERS)
    if resp.status_code != 200:
        logger.warning("GET %s %s -> %d", path, params, resp.status_code)
    return resp


def html_to_text(raw: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", raw)
    text = re.sub(r"</p>", "\n\n", text)
    text = re.sub(r"<script.*?</script>", "", text, flags=re.S)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_lib.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def section(title: str) -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info(title)
    logger.info("=" * 72)


def sample_category(name: str, cat_id: int, n: int = 30) -> list[dict]:
    """Pull a sample of posts from one category and characterise them."""
    resp = get("/wp/v2/posts", {
        "categories": cat_id, "per_page": n,
        "_fields": "id,slug,date,title,content,excerpt,ppma_author,authors,infotag,link",
    })
    if resp.status_code != 200:
        return []
    posts = resp.json()

    word_counts, empty, bylines, dates = [], 0, Counter(), []
    for p in posts:
        body = html_to_text(p.get("content", {}).get("rendered", ""))
        title = html_lib.unescape(p.get("title", {}).get("rendered", "")).strip()
        wc = len(body.split())
        word_counts.append(wc)
        if wc < 20:
            empty += 1
        for a in (p.get("authors") or []):
            bylines[a.get("display_name") or a.get("name") or "?"] += 1
        dates.append(p.get("date", "")[:7])
        del title

    logger.info("  %-24s n=%-3d  words: median=%-5s mean=%-6s min=%-4s max=%-6s  <20w: %d",
                name, len(posts),
                int(statistics.median(word_counts)) if word_counts else 0,
                int(statistics.mean(word_counts)) if word_counts else 0,
                min(word_counts) if word_counts else 0,
                max(word_counts) if word_counts else 0,
                empty)
    logger.info("      bylines: %s", dict(bylines.most_common(5)))
    logger.info("      months : %s", dict(Counter(dates).most_common(3)))
    return posts


def probe_category_quality() -> None:
    section("Per-category content quality (30-post sample each)")
    samples = {}
    for name, cid in CATEGORIES.items():
        samples[name] = sample_category(name, cid)
    (RAW / "category_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("  saved research/raw/category_samples.json")


def probe_uncategorized() -> None:
    section("What is in the 'Uncategorized @hy' bucket (59,751 posts)?")
    resp = get("/wp/v2/posts", {
        "categories": 1, "per_page": 10,
        "_fields": "id,slug,date,title,content,link",
    })
    for p in resp.json():
        body = html_to_text(p.get("content", {}).get("rendered", ""))
        title = html_lib.unescape(p.get("title", {}).get("rendered", "")).strip()
        logger.info("  %s | %s | %dw | %s", p["date"][:10], p["link"], len(body.split()), title[:70])


def probe_bogus_dates() -> None:
    section("Date quality (oldest post claimed 2000-01-01)")
    for before, label in (("2005-01-01T00:00:00", "before 2005"),
                          ("2015-01-01T00:00:00", "before 2015"),
                          ("2020-01-01T00:00:00", "before 2020")):
        resp = get("/wp/v2/posts", {"per_page": 1, "before": before})
        total = int(resp.headers.get("X-WP-Total", -1))
        logger.info("  posts dated %-12s : %s", label, f"{total:,}")

    resp = get("/wp/v2/posts", {
        "per_page": 5, "order": "asc", "orderby": "date",
        "_fields": "id,slug,date,title,link",
    })
    logger.info("  five oldest:")
    for p in resp.json():
        logger.info("      %s  %s  %s", p["date"], p["link"],
                    html_lib.unescape(p.get("title", {}).get("rendered", ""))[:60])


def probe_infocom_news_endpoint() -> None:
    section("Custom endpoint /infocom/v1/news")
    resp = get("/infocom/v1/news")
    logger.info("  status=%d content-type=%s", resp.status_code, resp.headers.get("content-type"))
    if resp.status_code == 200:
        try:
            data = resp.json()
        except json.JSONDecodeError:
            logger.warning("  not JSON: %s", resp.text[:300])
            return
        logger.info("  type=%s", type(data).__name__)
        if isinstance(data, list) and data:
            logger.info("  %d items, first item keys: %s", len(data), sorted(data[0].keys()))
            logger.info("  first item: %s", json.dumps(data[0], ensure_ascii=False)[:600])
        elif isinstance(data, dict):
            logger.info("  keys: %s", sorted(data.keys()))
            logger.info("  %s", json.dumps(data, ensure_ascii=False)[:600])
        (RAW / "infocom_v1_news.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2)[:200000], encoding="utf-8"
        )


def probe_excerpt_vs_content() -> None:
    section("Is `excerpt` a real summary or just truncated body?")
    resp = get("/wp/v2/posts", {"per_page": 5, "_fields": "id,content,excerpt"})
    for p in resp.json():
        body = html_to_text(p["content"]["rendered"])
        exc = html_to_text(p["excerpt"]["rendered"])
        prefix_match = body[:100] == exc[:100]
        logger.info("  post %-8s body=%-5dw excerpt=%-4dw  excerpt is body-prefix: %s",
                    p["id"], len(body.split()), len(exc.split()), prefix_match)


def probe_acf() -> None:
    section("ACF custom fields — anything useful?")
    resp = get("/wp/v2/posts", {"per_page": 3, "_fields": "id,acf,meta"})
    for p in resp.json():
        logger.info("  post %s acf=%s", p["id"],
                    json.dumps(p.get("acf"), ensure_ascii=False)[:300])
        logger.info("           meta=%s", json.dumps(p.get("meta"), ensure_ascii=False)[:300])


def main():
    probe_category_quality()
    probe_uncategorized()
    probe_bogus_dates()
    probe_excerpt_vs_content()
    probe_acf()
    probe_infocom_news_endpoint()
    logger.info("")
    logger.info("Done.")


if __name__ == "__main__":
    main()
