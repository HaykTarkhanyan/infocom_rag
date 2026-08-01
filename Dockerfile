# Single image running both processes: FastAPI privately on 127.0.0.1:8000 and
# Chainlit on 8001, wired together by start.sh. Caddy (see docker-compose.yml)
# terminates TLS in front of it.

FROM python:3.12-slim

# curl is used by start.sh to poll /health before the UI accepts traffic.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# IMPORTANT, and easy to get wrong: on Linux the default PyPI `torch` wheel
# bundles the CUDA runtime (nvidia-* packages, several GB). This box has no GPU,
# so torch is installed from PyTorch's CPU-only index FIRST, and the rest of the
# requirements afterwards -- by then torch is already satisfied and pip will not
# pull the CUDA build. On Windows the default wheel is already CPU-only, which is
# why this never came up in development.
COPY requirements.txt .
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.13.0" \
 && grep -v '^torch==' requirements.txt > /tmp/rest.txt \
 && pip install --no-cache-dir -r /tmp/rest.txt

COPY . .

# Model weights live in a mounted volume rather than the image: it keeps the
# image ~2 GB smaller, and the download survives rebuilds and restarts.
ENV HF_HOME=/models \
    HF_HUB_DISABLE_SYMLINKS=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    PORT=8001 \
    PYTHONUNBUFFERED=1

# Run as a non-root user; /models must be writable for the first-boot download.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /models /app/logs \
 && chown -R app:app /models /app
USER app

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
  CMD curl -sf http://127.0.0.1:8000/health || exit 1

CMD ["bash", "start.sh"]
