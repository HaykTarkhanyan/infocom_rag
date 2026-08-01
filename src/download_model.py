"""Download ATE-2 (Armenian Text Embeddings 2) weights from Hugging Face.

Credentials and paths come from .env -- nothing is hardcoded here.

Usage:
    python src/download_model.py --tokenizer-only --both   # ~21 MB, no weights
    python src/download_model.py                 # the model named in .env
    python src/download_model.py --both          # base and large, for comparison
    python src/download_model.py --model Metric-AI/armenian-text-embeddings-2-large

`--tokenizer-only` fetches the tokenizer and config without the weights (~21 MB
versus ~3.2 GB for both models). That is enough to build and verify chunking,
which depends on token counts rather than on the weights, so the expensive
download can wait for a better connection.

ATE-2 models (Metric-AI, MIT, arXiv 2603.22290):
    -base   278M params, 768-dim, finetuned from intfloat/multilingual-e5-base
    -large  560M params, 1024-dim, finetuned from intfloat/multilingual-e5-large

Both are XLM-R derivatives and share one tokenizer, so both still cap at 512
tokens. Measured on our cleaned corpus, Armenian runs 1.99 tokens/word through
it, putting 512 tokens at roughly 257 words. Our articles have a median of 3,001
tokens, and 90 of 94 exceed the cap -- chunking has to respect that rather than
rely on truncation.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing huggingface_hub: HF_HOME is read at import time,
# so setting it afterwards would be ignored.
load_dotenv()

HF_HOME = os.getenv("HF_HOME")
if HF_HOME:
    os.environ["HF_HOME"] = HF_HOME

# The HF cache symlinks blobs into snapshots, which needs admin rights or
# Developer Mode on Windows and otherwise dies with WinError 1314. Copying
# costs disk but works for every user.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("logs/download_model.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

ATE2_BASE = "Metric-AI/armenian-text-embeddings-2-base"
ATE2_LARGE = "Metric-AI/armenian-text-embeddings-2-large"

# Skip the PyTorch .bin duplicates when safetensors are present -- otherwise the
# download is roughly twice the necessary size.
IGNORE_PATTERNS = ["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.onnx_data"]

# Everything needed to tokenize, and nothing that holds weights.
TOKENIZER_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "*config.json",
]


def download(repo_id: str, token: str | None, tokenizer_only: bool = False) -> Path:
    logger.info("Downloading %s%s ...", repo_id, " (tokenizer only)" if tokenizer_only else "")
    try:
        path = snapshot_download(
            repo_id=repo_id,
            token=token,
            allow_patterns=TOKENIZER_PATTERNS if tokenizer_only else None,
            ignore_patterns=None if tokenizer_only else IGNORE_PATTERNS,
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            f"{repo_id} is gated and this token cannot access it. "
            "Accept the licence on the model page, then retry."
        ) from exc
    except RepositoryNotFoundError as exc:
        raise RuntimeError(
            f"{repo_id} not found. Check the name, or whether the token has access."
        ) from exc

    local = Path(path)
    size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
    logger.info("  -> %s", local)
    logger.info("  -> %d files, %.2f GB", sum(1 for f in local.rglob("*") if f.is_file()),
                size / 1024**3)
    return local


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ATE-2 embedding weights")
    parser.add_argument("--model", help="Repo id (default: EMBEDDING_MODEL from .env)")
    parser.add_argument("--both", action="store_true",
                        help="Download both -base and -large for benchmarking")
    parser.add_argument("--tokenizer-only", action="store_true",
                        help="Fetch tokenizer + config without weights (~21 MB for both)")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set in .env -- proceeding anonymously "
                       "(fine for public repos like ATE-2)")
    else:
        # Log only that a token exists, never any part of it. Partial tokens in
        # logs are a needless leak, and logs get pasted into issues and chats.
        logger.info("Using HF_TOKEN from .env (%d chars)", len(token))

    if HF_HOME:
        logger.info("HF_HOME = %s", HF_HOME)

    if args.both:
        repos = [ATE2_BASE, ATE2_LARGE]
    elif args.model:
        repos = [args.model]
    else:
        configured = os.getenv("EMBEDDING_MODEL")
        if not configured:
            logger.error("No model given: set EMBEDDING_MODEL in .env or pass --model")
            sys.exit(1)
        repos = [configured]

    for repo_id in repos:
        download(repo_id, token, tokenizer_only=args.tokenizer_only)

    if args.tokenizer_only:
        logger.info("Tokenizers only -- weights still need downloading before embedding.")
    logger.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)
