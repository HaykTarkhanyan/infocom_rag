"""Tests for article text extraction -- no network.

`extract_content` is shared by fetch_articles.py (indepth) and fetch_news.py
(news), deliberately: two extractors would drift and the drift would be
invisible. These tests pin the behaviours that a change to CHROME_SELECTORS or
BLOCK_SELECTOR could quietly break.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fetch_articles import extract_content


class TestBlockExtraction:
    def test_paragraphs_become_blocks(self):
        text, _ = extract_content("<p>Առաջին պարբերություն</p><p>Երկրորդ</p>")
        assert text == "Առաջին պարբերություն\n\nԵրկրորդ"

    def test_headings_are_marked(self):
        text, _ = extract_content("<h2>Վերնագիր</h2><p>Մարմին</p>")
        assert text.startswith("## Վերնագիր")

    def test_list_items_are_marked(self):
        text, _ = extract_content("<ul><li>Առաջին</li><li>Երկրորդ</li></ul>")
        assert "- Առաջին" in text and "- Երկրորդ" in text

    def test_consecutive_duplicate_blocks_collapse(self):
        """Nested page-builder markup re-emits the same text."""
        text, _ = extract_content("<p>Կրկնվող</p><p>Կրկնվող</p><p>Ուրիշ</p>")
        assert text == "Կրկնվող\n\nՈւրիշ"


class TestDivOnlyBodies:
    """Regression: a body with no block elements silently extracted to "".

    Measured on one month of `news`: 7 of 1236 posts (0.6%) carry their whole
    body as bare text inside <div>s. BLOCK_SELECTOR has no `div`, so nothing
    matched and the post became an empty string -- no error, no warning, just a
    document with no text. Zero of the 94 indepth articles hit this, which is why
    it survived until news was fetched.
    """

    def test_div_only_body_is_recovered(self):
        html = "<div>Սննդամթերքի անվտանգության տեսչական մարմինը հորդորում է։</div>"
        text, _ = extract_content(html)
        assert "Սննդամթերքի" in text
        assert text.strip(), "a div-only body must not extract to empty"

    def test_nested_divs_do_not_duplicate_text(self):
        """Why the fix is a fallback and not `div` in BLOCK_SELECTOR.

        Divs nest, so every ancestor would re-emit its descendants' text, and the
        dedup only collapses CONSECUTIVE identical blocks.
        """
        html = "<div><div><div>Եզակի նախադասություն։</div></div></div>"
        text, _ = extract_content(html)
        assert text.count("Եզակի նախադասություն") == 1

    def test_fallback_does_not_fire_when_blocks_exist(self):
        """The normal path must be untouched -- verified at 0/400 real records."""
        html = "<div><p>Իրական պարբերություն</p><div>chrome</div></div>"
        text, _ = extract_content(html)
        assert text == "Իրական պարբերություն"

    def test_genuinely_empty_body_stays_empty(self):
        assert extract_content("")[0] == ""
        assert extract_content("<div>   </div>")[0] == ""


class TestByline:
    def test_reads_author_link(self):
        html = '<a href="/author/some-name/">Անի Գրիգորյան</a><p>Մարմին</p>'
        _, byline = extract_content(html)
        assert byline == "Անի Գրիգորյան"

    def test_absent_byline_is_none(self):
        _, byline = extract_content("<p>Մարմին</p>")
        assert byline is None
