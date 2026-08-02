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

Memory is the binding constraint, not CPU — but **far less of one than we
thought**. Measured on the actual deployed container, Ubuntu 26.04 / Docker,
across four real queries:

```
after 1 query   782 MiB
after 2         785 MiB
after 3         788 MiB
after 4         792 MiB      <- stable, ~3 MiB drift per query
```

| retriever | Windows dev box | **measured on Linux** | needs |
|---|---|---|---|
| dense, ATE-2-**large** | 1793 MB | **~790 MB** | 2 GB is ample |
| dense, ATE-2-**base** | 958 MB | not re-measured | — |
| bm25 (no torch) | ~250 MB | not re-measured | anything |

**The 1793 MB figure was a Windows artifact and overstated the requirement by
2.3x.** The original note had the cause right — "torch's forward-pass arena, not
the weights, which load in 630 MB" — without noticing that an allocator arena is
precisely the thing that varies by platform. On Linux it is 630 MB of weights
plus ~160 MB of overhead, and it does not balloon.

That wrong number ruled out every 2 GB option in the provider comparison below.
It did not change the final choice (cx23 is the cheapest plan on offer at any
size; the 2 GB cpx12 costs *more*), but the reasoning that got there was wrong.
**Measure a deployment constraint on the deployment platform.**

Query latency on cx23's shared vCPU: **2.0-3.8s** end to end, retrieval plus
generation.

Most PaaS prices CPU generously and RAM stingily, which is why a raw VPS wins
here. **Read from the Hetzner console on 2026-08-02, Helsinki, incl. 19% VAT:**

| plan | | monthly | note |
|---|---|---|---|
| **cx23** (Cost-Optimized) | 2 vCPU / 4 GB | **€6.53 + €0.60 IPv4 = €7.13** | what we use |
| cpx22 (Regular Performance) | 2 vCPU / 4 GB | €23.19 + €0.60 = €23.79 | **3.3x more** |
| cx33 / cx43 / cx53 | 8 / 16 / 32 GB | €10.10 / €19.03 / €35.09 | sold out at HEL1 |

**Pick cx23 or an equally cheap ≥4 GB plan, and check the price before you buy.**
An earlier revision of this file told you to pick CPX22 "because the
cost-optimized line shows not available". That was a temporary stock-out, and
following the instruction would have cost **3.3x** for identical RAM — enough to
put Hetzner level with the $25/mo Render plan that decision #19 rejected for
being too expensive. Availability rotates; re-check both columns.

The trade is that you own the machine — OS patches, backups, and a bad deploy has
no rollback button.

---

## One-time setup

### 1. Create the server

Hetzner Cloud console → project → **Add Server**:

- **Image:** Ubuntu 26.04 LTS (the console default). Docker Engine officially
  supports it — verified against docs.docker.com, which lists Resolute 26.04,
  25.10, Noble 24.04 and Jammy 22.04. 24.04 also works if you prefer it.
- **Type:** **cx23** — 2 vCPU / 4 GB, under *Shared Resources → Cost-Optimized*.
  See the price table above before substituting anything.
- **SSH key:** add yours — do not use password auth. Hetzner emails a root
  password instead if you skip this.
- **Location:** Nuremberg / Falkenstein / Helsinki (eu-central), Singapore,
  Hillsboro or Ashburn. Singapore adds €8.33/mo; the EU three do not.
- **Name:** anything; `infocom-rag` keeps the console readable.

### 2. Point DNS at it — only if you have a domain

**Optional.** With no domain the stack serves plain HTTP on the IP (see below).

With one, create an `A` record (e.g. `rag.example.com`) → the server's IPv4 and
set `DOMAIN` in `.env`. **Do this before starting Caddy**: it validates the
domain over HTTP to issue the certificate, and repeated failures hit Let's
Encrypt rate limits.

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
                   # IMPORTANT: delete the HF_HOME line. It is a Windows dev path
                   # and would override the container's HF_HOME=/models volume.
docker compose up -d --build
```

`.env` on the server needs:

| key | why |
|---|---|
| `DOMAIN` | **optional.** Set → Caddy serves HTTPS with an auto-renewed Let's Encrypt certificate. Unset → plain HTTP on `:80` at the bare IP. Let's Encrypt will not certify an IP address, so a domain is the only route to HTTPS |
| `ACME_EMAIL` | Let's Encrypt expiry notices. Unused in HTTP mode |
| `OPENROUTER_API_KEY` | generation |
| `NEON_DB_STRING` | Chainlit persistence (threads, steps, feedback) |
| `CHAINLIT_AUTH_SECRET` | signs session cookies. **Required** whenever `APP_PASSWORD` is set — Chainlit refuses to start with auth enabled and no secret |
| `APP_PASSWORD` | **gates the UI** — see below |

### 5. Initialise the database once

```bash
docker compose exec app python src/db.py --init --check
```

### 6. Watch the first boot

The **first** start downloads ~2 GB of model weights into the `hf_models`
volume. Expect several minutes. `start.sh` waits for `/health` before starting
Chainlit, with a budget generous enough to cover that download
(`API_WAIT_SECONDS`, default 900) — a shorter one was measured expiring while
the model was still loading, putting the UI in front of a dead API.

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

## Running without a domain (plain HTTP)

Leave `DOMAIN` unset and Caddy listens on `:80` with no TLS. Reach the UI at
`http://YOUR_SERVER_IP/`. Verify what it resolved to:

```bash
docker compose config | grep SITE_ADDRESS      # ":80" = HTTP, a hostname = HTTPS
```

**What this costs you.** Everything is cleartext on the wire: the
`APP_PASSWORD`, every question, every answer. Anyone between the browser and the
server — a shared café network, a hotspot, an ISP — can read and alter it, and
the browser will mark the page "Not secure".

Deliberate for a short-lived demo, and **not acceptable once anyone else uses
it**. Use a throwaway `APP_PASSWORD` you do not reuse anywhere.

**The upgrade is one variable.** Point a domain's `A` record at the IP, put
`DOMAIN=` in `.env`, `docker compose up -d`. No rebuild, no image change — Caddy
requests the certificate on its own. A free `sslip.io` name (`157-90-1-2.sslip.io`
resolves to `157.90.1.2`) works if you do not want to register anything.

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
- **Built and run locally** on 2026-08-02: image 1.44 GB, container memory 1.83
  GiB — matching the 1793 MB measured on the host, which is what validates the
  sizing table above. Doing so surfaced two startup bugs that reading, linting
  and 15 passing tests had all missed. Full boot (`/health` answering, then a
  real question) is still **unverified** — see the session log.
- Testing locally, mount a **native docker volume**, not a Windows host path.
  Bind-mounting the Windows model cache is unusably slow through Docker
  Desktop's filesystem translation (still loading after 15 minutes), and
  `docker stats` then reads high because it counts page cache.

---

## Appendix: Render

`render.yaml`, and the `plan: standard` note inside it, still work — one service
running `start.sh`, secrets added once to a shared env group. It costs $25/mo for
2 GB, which puts ATE-2-large at 88% RSS with no headroom for FastAPI, Chainlit
and the OS. Viable with ATE-2-base; tight with large.
