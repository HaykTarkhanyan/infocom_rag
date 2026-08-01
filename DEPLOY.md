# Deploying

Target: a **Hetzner Cloud VPS** running Docker, with Caddy terminating TLS.
Render config is kept as an alternative in the appendix.

```
Hetzner VPS (4 GB)
└── docker compose
    ├── caddy   :80 :443   automatic Let's Encrypt, reverse proxy
    └── app     :8001
        ├── uvicorn api:app   127.0.0.1:8000   (private)
        └── chainlit          0.0.0.0:8001     (public, behind Caddy)
```

Chainlit talks to the API over loopback rather than importing the pipeline, so
**what is deployed is the path the eval measures**.

---

## Why this, and what it costs

Memory is the binding constraint, not CPU. Peak RSS is **sustained** — it
plateaus after the first query and is never released:

| retriever | peak RSS | needs |
|---|---|---|
| dense, ATE-2-**large** | **1793 MB** | ≥ 4 GB box |
| dense, ATE-2-**base** | 958 MB | 2 GB is workable |
| bm25 (no torch) | ~250 MB | anything |

Thread count does not help (1791 MB at 1 thread vs 1792 at 4) — the memory is
torch's forward-pass arena, not the weights, which load in 630 MB.

Most PaaS prices CPU generously and RAM stingily, which is why a raw VPS wins
here: **2 vCPU / 4 GB for roughly €6/mo** versus $25/mo for 2 GB on Render. The
trade is that you own the machine — OS patches, backups, and a bad deploy has no
rollback button.

> Prices move. Hetzner raised cloud prices in June 2026 and their shared-vCPU
> tier periodically shows "currently not available" by region. Check the console
> before assuming a figure. Pick any plan with **≥ 4 GB RAM**.

---

## One-time setup

### 1. Create the server

Hetzner Cloud console → new project → **Add Server**:

- **Image:** Ubuntu 24.04
- **Type:** shared vCPU, **≥ 4 GB RAM** (CX-class)
- **SSH key:** add yours — do not use password auth
- Location: whichever is nearest your users

### 2. Point DNS at it

Create an `A` record for your domain (e.g. `rag.example.com`) → the server's
IPv4. **Do this before starting Caddy**: it validates the domain over HTTP to
issue a certificate, and repeated failures hit Let's Encrypt rate limits.

### 3. Harden and install Docker

```bash
ssh root@YOUR_SERVER_IP

apt update && apt upgrade -y
apt install -y ufw
ufw allow OpenSSH && ufw allow 80 && ufw allow 443 && ufw --force enable

curl -fsSL https://get.docker.com | sh
```

### 4. Deploy

```bash
git clone https://github.com/HaykTarkhanyan/infocom_rag.git
cd infocom_rag
cp .env.example .env
nano .env          # fill in the values below, then:
docker compose up -d --build
```

`.env` on the server needs:

| key | why |
|---|---|
| `DOMAIN` | e.g. `rag.example.com` — Caddy issues the certificate for it |
| `ACME_EMAIL` | Let's Encrypt expiry notices |
| `OPENROUTER_API_KEY` | generation |
| `NEON_DB_STRING` | Chainlit persistence (threads, steps, feedback) |
| `CHAINLIT_AUTH_SECRET` | any long random string; signs session cookies |
| `APP_PASSWORD` | **gates the UI** — see below |

### 5. Initialise the database once

```bash
docker compose exec app python src/db.py --init --check
```

### 6. Watch the first boot

The **first** start downloads ~2 GB of model weights into the `hf_models`
volume. Expect several minutes. `start.sh` polls `/health` for up to 60s before
starting Chainlit, so if the download runs longer the UI comes up on the next
restart — which `restart: unless-stopped` handles.

```bash
docker compose logs -f app
curl -sI https://YOUR_DOMAIN        # expect 200 once Caddy has a certificate
```

---

## `APP_PASSWORD` is not optional

There is **no per-user cost cap and no rate limiting**. Anyone who finds the URL
can spend your OpenRouter key. `chainlit_app.py` registers a password gate only
when `APP_PASSWORD` is set, so local development stays frictionless and the
public deployment does not.

This is a shared password — a lock on the door, not identity.

---

## Operating it

```bash
docker compose logs -f app            # follow logs
docker compose restart app            # restart without rebuilding
docker compose up -d --build          # deploy new code
docker stats --no-stream              # confirm the app sits near 1.8 GB
docker compose exec app python src/db.py --check     # rows landing in Neon?
```

**Updating:** `git pull && docker compose up -d --build`. The model volume
persists, so rebuilds do not re-download 2 GB.

**Memory:** the app container is capped at 3 GB (`mem_limit`). If a leak ever
develops, Docker kills the container rather than the host, and it restarts.

**Backups:** Hetzner backups are a paid add-on. Little here is irreplaceable —
code is in git, the model re-downloads, chat history is in Neon. The one thing
worth keeping is `caddy_data` (certificates); losing it forces re-issuance and
can hit rate limits.

**Patching:** `apt update && apt upgrade` periodically, or enable
`unattended-upgrades`. This is the ongoing cost of the cheap plan — an unpatched
internet-facing box is a real liability.

---

## Changing the model or retriever

Edit `config.toml`, then redeploy. Two coupled changes people forget:

- `max_distance` is **model-specific**. 0.55 was tuned for ATE-2-large; base's
  equivalent is roughly 0.30. Changing the model without changing this silently
  disables filtering.
- The vector index must be rebuilt for the new model
  (`src/embed_corpus_colab.py`) and committed.

Switching to `retriever = "bm25"` removes the need for torch entirely and drops
peak memory to ~250 MB — at a measurable cost in retrieval quality.

---

## Notes on the build

- **`torch` is installed from PyTorch's CPU index, deliberately.** On Linux the
  default PyPI wheel bundles the CUDA runtime — several GB of `nvidia-*`
  packages this box cannot use. On Windows the default is already CPU-only,
  which is why the problem does not appear in development.
- Model weights live in a **volume**, not the image: it keeps the image ~2 GB
  smaller and survives rebuilds.
- The container runs as a non-root user.
- The image was **not** built locally (no Docker daemon on the dev machine), so
  the first build on the server is its first real test.

---

## Appendix: Render

`render.yaml`, and the `plan: standard` note inside it, still work — one service
running `start.sh`, secrets added once to a shared env group. It costs $25/mo for
2 GB, which puts ATE-2-large at 88% RSS with no headroom for FastAPI, Chainlit
and the OS. Viable with ATE-2-base; tight with large.
