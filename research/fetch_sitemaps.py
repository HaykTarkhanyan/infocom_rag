"""Fetch and summarize infocom.am sitemaps into research/raw/."""

import logging
import re
from pathlib import Path

import requests

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("logs/fetch_sitemaps.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

BASE = "https://infocom.am"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; infocom_rag)"}
RAW = Path("research/raw")
RAW.mkdir(parents=True, exist_ok=True)


def get(url: str) -> requests.Response:
    resp = requests.get(url, timeout=30, headers=HEADERS)
    logger.info("GET %s -> %d (%d bytes, %s)", url, resp.status_code,
                len(resp.content), resp.headers.get("content-type", "?"))
    return resp


def is_xml(resp: requests.Response) -> bool:
    """A real sitemap is XML, not a WordPress HTML 404 page."""
    return "xml" in resp.headers.get("content-type", "") and resp.text.lstrip().startswith("<?xml")


def save(name: str, text: str) -> None:
    (RAW / name).write_text(text, encoding="utf-8")
    logger.info("  saved research/raw/%s", name)


def main():
    candidates = ["sitemap_index.xml", "news-sitemap.xml", "wp-sitemap.xml", "sitemap.xml"]
    live = {}

    for name in candidates:
        resp = get(f"{BASE}/{name}")
        if resp.status_code == 200 and is_xml(resp):
            save(name, resp.text)
            live[name] = resp.text
        else:
            logger.warning("  NOT a usable sitemap (status=%d, xml=%s)",
                           resp.status_code, is_xml(resp))

    # Expand any sitemap index into its child sitemaps
    for name, text in list(live.items()):
        children = re.findall(r"<sitemap>.*?<loc>(.*?)</loc>", text, re.S)
        if not children:
            continue
        logger.info("%s is an index with %d child sitemaps:", name, len(children))
        for child in children:
            logger.info("    %s", child)

    # Summarize URL counts per sitemap
    for name, text in live.items():
        urls = re.findall(r"<url>.*?<loc>(.*?)</loc>", text, re.S)
        logger.info("%s contains %d <url> entries", name, len(urls))
        for u in urls[:5]:
            logger.info("    sample: %s", u)


if __name__ == "__main__":
    main()
