"""Download the embedding model at BUILD time, not first request.

Without this the first request after a deploy pays for a ~2 GB download, which
blows past any sensible health-check timeout and makes the service look broken.
Render caches the build filesystem, so this runs once per deploy rather than per
boot.

Reads the model name from config.toml, so it cannot drift from what the app
actually loads.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from config import settings


def main() -> None:
    model = settings.embedding.model
    print(f"Prefetching {model} into {os.environ.get('HF_HOME', '(default cache)')}")

    if settings.retrieval.retriever != "dense":
        print(f"retriever is {settings.retrieval.retriever!r}, not 'dense' -- "
              "skipping model download")
        return

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=model,
        ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.onnx_data"],
    )
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    print(f"Cached {total / 1024**3:.2f} GB at {path}")


if __name__ == "__main__":
    main()
