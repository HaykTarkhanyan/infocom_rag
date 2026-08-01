# Deploying to Render

One web service runs both processes: FastAPI privately on `127.0.0.1:8000`,
Chainlit publicly on `$PORT`. Same shape as the Washington project, adapted for
Chainlit instead of Streamlit.

```
Render web service
├── uvicorn api:app        127.0.0.1:8000   (private)
└── chainlit run           0.0.0.0:$PORT    (public)
        └── RAG_API_URL=http://127.0.0.1:8000
```

Chainlit talks to the API over loopback rather than importing the pipeline, so
**what is deployed is the same path the eval measured**.

---

## The one thing that decides your bill

The embedding model runs in-process, and its memory is the dominant cost.
Measured peak RSS, **sustained** (it plateaus after the first query and is never
released):

| retriever | peak RSS | smallest Render plan | retrieval quality |
|---|---|---|---|
| dense, ATE-2-**large** | **1793 MB** | Standard 2 GB — 90% used | best: 5/5 relevant in top-5 |
| dense, ATE-2-**base** | **958 MB** | Standard 2 GB — comfortable | 2/5 relevant in top-5 |
| **bm25** (no torch) | ~250 MB | Free / Starter 512 MB | 3/5, and the noise made the model decline |

Notes that are easy to get wrong:

- **Thread count does not help.** 1791 MB at 1 thread vs 1792 MB at 4. The
  memory is torch's forward-pass arena, not the weights (which load in 630 MB).
- **The free tier spins down after 15 minutes.** A cold start would reload a 2 GB
  model, so the first visitor after idle waits minutes. Free is not viable with
  dense retrieval regardless of the RAM ceiling.
- **large on Standard leaves ~200 MB** for FastAPI, Chainlit, Python and the OS.
  That is tight enough to expect OOM kills. Either take the next tier up, or run
  base.
- Hugging Face serverless inference **cannot** host ATE-2 (`410: deprecated and
  no longer supported by provider hf-inference`), so moving query embedding off
  the box would need a dedicated Inference Endpoint.

To switch retriever or model, edit `config.toml` — `[retrieval] retriever` and
`[embedding] model` — and redeploy. If you switch to `bm25`, remove `torch` from
`requirements.txt` too, or the build still downloads it.

**If you change `[embedding] model`, `max_distance` must change with it.** It is
model-specific: 0.55 was tuned for large; base's equivalent is around 0.30.
`data/vectors_large.npz` also has to be rebuilt for the other model.

---

## Setup

### 1. Commit the data files

`data/chunks.jsonl` (3.7 MB) and `data/vectors_large.npz` (3.6 MB) are committed
deliberately — Render builds from git, and the vectors cannot be rebuilt there
(no GPU, and CPU indexing would take ~46 min inside a build step). Everything
else under `data/` stays ignored.

### 2. Push and create the Blueprint

Render reads `render.yaml`. Point a new Blueprint at the repo.

### 3. Add the secrets ONCE

Render rejects `sync: false` inside an `envVarGroups` block, so secrets are not
in the YAML. After the first sync, open the **`infocom-rag-shared`** group in the
dashboard and add:

| key | what it is |
|---|---|
| `OPENROUTER_API_KEY` | generation and the eval judge |
| `NEON_DB_STRING` | Chainlit's data layer (threads, steps, feedback) |
| `CHAINLIT_AUTH_SECRET` | any long random string; signs session cookies |
| `APP_PASSWORD` | **gates the UI.** Without it the service is open to anyone |

`HF_TOKEN` is **not** needed at runtime — ATE-2 is public and the weights are
pulled during build.

### 4. Initialise the database once

The Chainlit tables must exist before first boot. From your machine, with
`NEON_DB_STRING` in `.env`:

```bash
python src/db.py --init --check
```

---

## Why `APP_PASSWORD` is not optional

There is **no per-user cost cap and no rate limiting**. Anyone who finds the URL
can spend your OpenRouter key; at ~$0.005 a question that is slow to hurt, but it
is unbounded. `chainlit_app.py` registers a password gate only when
`APP_PASSWORD` is set, so local development stays frictionless and the deployed
service does not.

This is a shared password, not real auth. It is a lock on the door, not identity.

---

## Build details

- `requirements.txt` is generated from `pyproject.toml` (Render builds with pip,
  not uv). Regenerate after changing dependencies:

  ```bash
  python -c "import tomllib;d=tomllib.load(open('pyproject.toml','rb'));print('\n'.join(sorted(d['project']['dependencies'])))"
  ```

- `scripts/prefetch_model.py` runs in the **build** step so the model is cached
  in the image. Without it the first request pays for a 2 GB download and the
  health check times out.
- `start.sh` polls `/health` for up to 60s before starting Chainlit. Loading the
  model takes ~15s and the UI is useless until the API answers.

## After deploying

```bash
curl -s https://<service>.onrender.com/            # Chainlit UI
python src/db.py --check                            # rows arriving in Neon?
python research/llm_cost_report.py --by day         # spend
```

The cost ledger (`logs/llm_calls.jsonl`) is **per-container and ephemeral** on
Render — the filesystem does not survive a redeploy. Cost telemetry that has to
persist lives in Neon, in `steps.metadata`, written by the UI. Query it with
`src/db.py --check`.
