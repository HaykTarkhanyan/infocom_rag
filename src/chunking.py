"""Split articles into embeddable chunks that fit ATE-2's 512-token window.

This is the piece the archived prototype got wrong. It embedded whole articles
with `truncation=True`, so on our corpus 90 of 94 articles (96%) would have had
their tail silently dropped from the vector while the full text was still handed
to the LLM. Nothing here truncates: text that does not fit is split, and the one
case that cannot be split (a single sentence over budget) is logged loudly.

Strategy, in order of preference:
  1. Split on `## ` headings, which `fetch_articles.py` preserves from <h2>.
  2. Within a section, greedily pack paragraphs up to the token budget.
  3. A paragraph over budget is split on sentence boundaries.
  4. A sentence over budget is split on token boundaries, with a warning.

Every chunk is prefixed with the article title and its section heading, so a
retrieved chunk is interpretable on its own and carries the topic words that
make it findable. That prefix is counted against the budget, not bolted on
afterwards.

Usage:
    python src/chunking.py                          # chunk data/articles.jsonl
    python src/chunking.py --max-tokens 480 --overlap-sentences 2
"""

import argparse
import json
import logging
import os
import re
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv

# load_dotenv puts HF_HOME into os.environ, which transformers reads at import
# time -- so this has to run before the transformers import below.
load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from transformers import AutoTokenizer

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("logs/chunking.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

DEFAULT_INPUT = "data/articles.jsonl"
DEFAULT_OUTPUT = "data/chunks.jsonl"

# e5-family models expect these prefixes. ATE-2 inherits the convention.
PASSAGE_PREFIX = "passage: "

# Armenian sentence terminators: վերջակետ (։), plus Latin punctuation that
# appears in numbers, abbreviations and quoted foreign text.
SENTENCE_SPLIT = re.compile(r"(?<=[։.!?])\s+")

HEADING_RE = re.compile(r"^## (.+)$")


def token_length(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def split_sections(text: str) -> list[tuple[str | None, list[str]]]:
    """Split article text into (heading, paragraphs) sections on `## ` lines."""
    sections: list[tuple[str | None, list[str]]] = []
    heading: str | None = None
    paragraphs: list[str] = []

    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        match = HEADING_RE.match(block)
        if match:
            if paragraphs:
                sections.append((heading, paragraphs))
                paragraphs = []
            heading = match.group(1).strip()
        else:
            paragraphs.append(block)

    if paragraphs:
        sections.append((heading, paragraphs))
    return sections


def split_oversized(tokenizer, text: str, budget: int, where: str) -> list[str]:
    """Split a paragraph that exceeds *budget* into pieces that fit.

    Tries sentence boundaries first. A single sentence over budget is split on
    token boundaries -- unavoidable, but logged rather than silently truncated.
    """
    pieces: list[str] = []
    current: list[str] = []

    for sentence in SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue

        if token_length(tokenizer, sentence) > budget:
            if current:
                pieces.append(" ".join(current))
                current = []
            logger.warning(
                "Sentence exceeds the %d-token budget in %s; splitting mid-sentence "
                "(%d tokens)", budget, where, token_length(tokenizer, sentence)
            )
            ids = tokenizer(sentence, add_special_tokens=False)["input_ids"]
            for start in range(0, len(ids), budget):
                pieces.append(tokenizer.decode(ids[start:start + budget]).strip())
            continue

        candidate = current + [sentence]
        if token_length(tokenizer, " ".join(candidate)) > budget:
            pieces.append(" ".join(current))
            current = [sentence]
        else:
            current = candidate

    if current:
        pieces.append(" ".join(current))
    return [p for p in pieces if p.strip()]


def pack_section(tokenizer, paragraphs: list[str], budget: int,
                 overlap_sentences: int, where: str) -> list[str]:
    """Greedily pack paragraphs into chunks under *budget*, with sentence overlap."""
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            chunks.append("\n\n".join(current))

    for paragraph in paragraphs:
        if token_length(tokenizer, paragraph) > budget:
            flush()
            current = []
            chunks.extend(split_oversized(tokenizer, paragraph, budget, where))
            continue

        candidate = current + [paragraph]
        if token_length(tokenizer, "\n\n".join(candidate)) > budget:
            flush()
            # Carry the tail of the previous chunk so a thought split across a
            # boundary is still retrievable from either side.
            tail: list[str] = []
            if overlap_sentences and current:
                sentences = [s for s in SENTENCE_SPLIT.split(current[-1]) if s.strip()]
                tail = sentences[-overlap_sentences:]
            seed = [" ".join(tail)] if tail else []
            if seed and token_length(tokenizer, "\n\n".join(seed + [paragraph])) <= budget:
                current = seed + [paragraph]
            else:
                current = [paragraph]
        else:
            current = candidate

    flush()
    return chunks


def chunk_article(article: dict, tokenizer, max_tokens: int,
                  overlap_sentences: int) -> list[dict]:
    """Split one article into chunks that fit within *max_tokens* once embedded."""
    title = article["title"]
    chunks: list[dict] = []

    for section_index, (heading, paragraphs) in enumerate(split_sections(article["text"])):
        # The context prefix is counted against the budget, not added on top of it.
        header = f"{PASSAGE_PREFIX}{title}"
        if heading and heading != title:
            header += f"\n{heading}"
        header += "\n\n"

        budget = max_tokens - token_length(tokenizer, header)
        if budget <= 0:
            raise ValueError(
                f"Title and heading alone exceed {max_tokens} tokens for post "
                f"{article['post_id']}; cannot build a chunk around them"
            )

        where = f"post {article['post_id']} section {section_index}"
        for part_index, body in enumerate(
            pack_section(tokenizer, paragraphs, budget, overlap_sentences, where)
        ):
            text = header + body
            chunks.append({
                "chunk_id": f"{article['post_id']}-{section_index}-{part_index}",
                "post_id": article["post_id"],
                "url": article["url"],
                "title": title,
                "heading": heading,
                "section_index": section_index,
                "part_index": part_index,
                "text": text,
                "n_tokens": token_length(tokenizer, text),
                "published": article["published"],
                "authors": article["authors"],
                "page_byline": article.get("page_byline"),
                "categories": article["categories"],
                "infotags": article["infotags"],
            })

    return chunks


def report(chunks: list[dict], articles: list[dict], max_tokens: int) -> None:
    if not chunks:
        logger.error("No chunks produced.")
        return

    lengths = sorted(c["n_tokens"] for c in chunks)
    over = [c for c in chunks if c["n_tokens"] > max_tokens]
    per_article: dict[int, int] = {}
    for chunk in chunks:
        per_article[chunk["post_id"]] = per_article.get(chunk["post_id"], 0) + 1

    logger.info("")
    logger.info("=" * 68)
    logger.info("Chunked %d articles into %d chunks", len(articles), len(chunks))
    logger.info("=" * 68)
    logger.info("  tokens/chunk: median=%d mean=%d min=%d max=%d",
                lengths[len(lengths) // 2], int(statistics.mean(lengths)),
                lengths[0], lengths[-1])
    logger.info("  chunks/article: median=%d max=%d",
                int(statistics.median(list(per_article.values()))),
                max(per_article.values()))
    logger.info("  budget utilisation: %.0f%% of %d tokens on average",
                100 * statistics.mean(lengths) / max_tokens, max_tokens)
    logger.info("  chunks OVER budget: %d  (must be 0)", len(over))
    logger.info("  chunks with a section heading: %d of %d",
                sum(1 for c in chunks if c["heading"]), len(chunks))

    if over:
        for chunk in over[:5]:
            logger.error("  OVER: %s %d tokens", chunk["chunk_id"], chunk["n_tokens"])
        raise SystemExit("Chunks exceed the token budget -- refusing to write a bad corpus")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk articles for embedding")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Hard token ceiling per chunk (default: ATE-2's 512)")
    parser.add_argument("--overlap-sentences", type=int, default=1,
                        help="Sentences carried across a chunk boundary (default: 1)")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL"),
                        help="Tokenizer to measure with (default: EMBEDDING_MODEL from .env)")
    args = parser.parse_args()

    if not args.model:
        logger.error("No tokenizer: set EMBEDDING_MODEL in .env or pass --model")
        sys.exit(1)

    source = Path(args.input)
    if not source.exists():
        logger.error("Input not found: %s -- run src/fetch_articles.py first", source)
        sys.exit(1)

    logger.info("Tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    articles = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()]
    logger.info("Loaded %d articles from %s", len(articles), source)

    chunks: list[dict] = []
    for article in articles:
        chunks.extend(chunk_article(article, tokenizer, args.max_tokens,
                                    args.overlap_sentences))

    report(chunks, articles, args.max_tokens)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info("")
    logger.info("Wrote %d chunks to %s", len(chunks), output)


if __name__ == "__main__":
    main()
