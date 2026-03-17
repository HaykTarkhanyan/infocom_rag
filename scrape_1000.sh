#!/usr/bin/env bash
# Quick-start script to scrape 1000 articles from infocom.am
# Run this on a machine with unrestricted internet access.
#
# Usage:
#   chmod +x scrape_1000.sh
#   ./scrape_1000.sh
#
set -euo pipefail

echo "=== Scraping 1000 articles from infocom.am ==="
echo "Using 8 parallel workers for speed..."

python3 web_scraper.py \
    --limit 1000 \
    --workers 8 \
    --output articles.jsonl \
    --resume

echo ""
echo "=== Done! ==="
wc -l articles.jsonl
echo "articles written to articles.jsonl"
