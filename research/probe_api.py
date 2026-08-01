"""Probe the infocom.am WordPress REST API and dump findings to research/raw/.

Answers the questions that decide the RAG design:
  - how many posts exist, per language (the site runs WPML)
  - who the real author is (the site runs PublishPress Authors)
  - which custom post types carry usable text
  - what the custom `infocom/v1` and `custom-api/v1` namespaces expose
  - how far back the archive goes
"""

import json
import logging
from pathlib import Path

import requests

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("logs/probe_api.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

API = "https://infocom.am/wp-json"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; infocom_rag)"}
RAW = Path("research/raw")
RAW.mkdir(parents=True, exist_ok=True)


def get(path: str, params: dict | None = None) -> requests.Response:
    resp = requests.get(f"{API}{path}", params=params, timeout=30, headers=HEADERS)
    logger.info("GET %s %s -> %d", path, params or "", resp.status_code)
    return resp


def count_of(path: str, params: dict | None = None) -> tuple[int, int]:
    """Return (total_items, total_pages) from WP pagination headers."""
    p = dict(params or {})
    p["per_page"] = 1
    resp = get(path, p)
    if resp.status_code != 200:
        logger.warning("  -> non-200, no counts available")
        return -1, -1
    total = int(resp.headers.get("X-WP-Total", -1))
    pages = int(resp.headers.get("X-WP-TotalPages", -1))
    return total, pages


def section(title: str) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)


def probe_languages() -> None:
    section("WPML: post counts per language")
    for lang in ("hy", "en", "ru", "all"):
        total, pages = count_of("/wp/v2/posts", {"lang": lang})
        logger.info("  lang=%-4s posts=%s pages=%s", lang, f"{total:,}", pages)


def probe_post_types() -> None:
    section("Post-type volumes")
    for base in ("posts", "pages", "reports", "rewards", "glossary", "media"):
        total, _ = count_of(f"/wp/v2/{base}")
        logger.info("  %-10s %s items", base, f"{total:,}")


def probe_authors() -> None:
    section("Authorship: WP user vs PublishPress author")
    resp = get("/wp/v2/posts", {"per_page": 3, "_embed": "author,wp:term"})
    posts = resp.json()
    for p in posts:
        wp_user = p.get("_embedded", {}).get("author", [{}])[0].get("name")
        terms = p.get("_embedded", {}).get("wp:term", [])
        ppma = []
        for group in terms:
            for t in group:
                if t.get("taxonomy") == "author":
                    ppma.append(t.get("name"))
        logger.info("  post %s slug=%s", p["id"], p.get("slug"))
        logger.info("      wp author field : %s", wp_user)
        logger.info("      ppma_author terms: %s", ppma or "(none embedded)")

    total, _ = count_of("/wp/v2/ppma_author")
    logger.info("  ppma_author taxonomy terms: %s", total)
    resp = get("/wp/v2/ppma_author", {"per_page": 20})
    if resp.status_code == 200:
        names = [t.get("name") for t in resp.json()]
        logger.info("  sample authors: %s", names)

    total, _ = count_of("/wp/v2/users")
    logger.info("  wp users (old scraper used these): %s", total)


def probe_custom_namespaces() -> None:
    section("Custom REST namespaces")
    for ns in ("/infocom/v1", "/custom-api/v1", "/wpml/v1"):
        resp = get(ns)
        if resp.status_code == 200:
            routes = list(resp.json().get("routes", {}).keys())
            logger.info("  %s exposes %d routes:", ns, len(routes))
            for r in routes:
                logger.info("      %s", r)
            (RAW / f"namespace{ns.replace('/', '_')}.json").write_text(
                json.dumps(resp.json(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            logger.warning("  %s -> %d", ns, resp.status_code)


def probe_archive_depth() -> None:
    section("Archive depth (oldest and newest posts)")
    for order, label in (("desc", "newest"), ("asc", "oldest")):
        resp = get("/wp/v2/posts", {"per_page": 1, "order": order,
                                    "orderby": "date", "_fields": "id,slug,date,link,title"})
        if resp.status_code == 200 and resp.json():
            p = resp.json()[0]
            logger.info("  %-7s id=%s slug=%s date=%s", label, p["id"], p.get("slug"), p["date"])
            logger.info("          %s", p.get("link"))


def probe_taxonomy_sizes() -> None:
    section("Taxonomy sizes")
    for base in ("categories", "tags", "infotag"):
        total, _ = count_of(f"/wp/v2/{base}")
        logger.info("  %-12s %s terms", base, f"{total:,}")
    resp = get("/wp/v2/categories", {"per_page": 100, "_fields": "id,name,slug,count,parent"})
    if resp.status_code == 200:
        cats = sorted(resp.json(), key=lambda c: -c["count"])
        logger.info("  categories by volume:")
        for c in cats:
            logger.info("      %-40s id=%-6s count=%-8s parent=%s",
                        f"{c['name']} ({c['slug']})", c["id"], f"{c['count']:,}", c["parent"])
        (RAW / "categories.json").write_text(
            json.dumps(cats, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def probe_content_shape() -> None:
    section("Content shape of a single post (what the API actually returns)")
    resp = get("/wp/v2/posts", {"per_page": 1})
    post = resp.json()[0]
    (RAW / "sample_post_full.json").write_text(
        json.dumps(post, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("  top-level keys: %s", sorted(post.keys()))
    content = post.get("content", {}).get("rendered", "")
    logger.info("  content.rendered: %d chars, protected=%s",
                len(content), post.get("content", {}).get("protected"))
    logger.info("  excerpt.rendered: %d chars",
                len(post.get("excerpt", {}).get("rendered", "")))
    logger.info("  saved full sample to research/raw/sample_post_full.json")


def main():
    probe_languages()
    probe_post_types()
    probe_archive_depth()
    probe_taxonomy_sizes()
    probe_authors()
    probe_custom_namespaces()
    probe_content_shape()
    logger.info("")
    logger.info("Done.")


if __name__ == "__main__":
    main()
