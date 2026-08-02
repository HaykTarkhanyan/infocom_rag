#!/usr/bin/env bash
# Render entrypoint.
#
# One web service runs both processes: FastAPI privately on 127.0.0.1:8000 and
# Chainlit publicly on $PORT. Chainlit reaches the API over loopback via
# RAG_API_URL, which is the same boundary the eval harness uses -- so what is
# deployed is the path that was measured.
#
# Two services would double the memory, and the embedding model is the dominant
# cost here (see DEPLOY.md).
set -euo pipefail

export RAG_API_URL="http://127.0.0.1:8000"

# Keep torch to one thread. It does NOT reduce peak memory (measured: 1791 MB at
# 1 thread vs 1792 MB at 4), but Render's Standard instance is 1 CPU and extra
# threads only add contention.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

# Wait for the API before accepting traffic -- the UI is useless until it answers,
# and starting early greets the first visitor with "cannot reach the API".
#
# The budget is generous on purpose. On a FIRST deploy the API has to download
# ~2 GB of model weights before it can even load them, and loading ATE-2-large
# itself is not instant. A 60s budget was measured expiring while the model was
# still loading, so the UI came up in front of a dead API.
API_WAIT_SECONDS="${API_WAIT_SECONDS:-900}"

echo "Waiting up to ${API_WAIT_SECONDS}s for the API (first boot downloads the model)..."
for i in $(seq 1 "$API_WAIT_SECONDS"); do
  if curl -sf --max-time 5 "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
    echo "API ready after ${i}s"
    break
  fi
  # Fail fast if uvicorn exited rather than burning the whole budget.
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "API process died during startup" >&2
    exit 1
  fi
  if [ $((i % 30)) -eq 0 ]; then
    echo "  still waiting (${i}s)..."
  fi
  sleep 1
done

if ! curl -sf --max-time 5 "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
  echo "API did not become healthy within ${API_WAIT_SECONDS}s; starting the UI anyway" >&2
fi

exec chainlit run chainlit_app.py \
  --host 0.0.0.0 \
  --port "${PORT:-8001}" \
  --headless
