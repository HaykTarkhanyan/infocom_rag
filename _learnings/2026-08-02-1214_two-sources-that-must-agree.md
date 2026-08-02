# Any value that must match in two places will eventually drift

Three instances of the same bug shape appeared in this project within two days.
Each time, two artefacts had to agree, nothing enforced it, and the failure mode
was silent degradation rather than an error.

| the two things | what breaks if they drift | guarded? |
|---|---|---|
| chunk boundaries vs the vector index | queries and passages land in different spaces | **yes** — `_dense_index()` refuses to load on a model mismatch or unknown chunk ids |
| `src/embedding.py` vs `src/embed_corpus_colab.py` pooling math | the index is built differently from how queries are embedded | **yes** — `research/verify_embedding_parity.py` asserts cosine ≥ 0.999 |
| `config.toml` vs `.env EMBEDDING_MODEL` | chunking tokenizes with one model, retrieval embeds with another | **no** — and this is the one that drifted |

The two that were guarded never broke. The one that was not, did:
`config.toml` said `...-2-large` while `.env` said `...-2-base`, so
`python src/chunking.py` measured token budgets with base while retrieval
embedded with large.

**Why it went unnoticed.** ATE-2 base and large ship a byte-identical
`tokenizer.json`, so the two happened to produce the same chunks. The system was
correct by coincidence. Point `config.toml` at any model with a different
tokenizer and it silently degrades — no exception, no failing test, just worse
retrieval that looks like a bad model.

**Fix.** Collapse to one source. `config.toml` is authoritative;
`chunking.py` and `download_model.py` now read `settings.embedding.model`, and
`EMBEDDING_MODEL` is gone from `.env`.

**Verified operationally, not by assertion**: re-chunking with the corrected
default produced a byte-identical file (`sha256 c20134d8825c5e4c`). That also
confirmed the shared-tokenizer claim first-hand, where previously it rested on
hashing `tokenizer.json`.

## The rule

When a value must be identical in two places, do one of:

1. **Collapse it to one place.** Always the first choice.
2. **Add a check that screams when they drift** — a load-time assertion, a parity
   test, a refusal to start.

A comment asserting the two match is worth nothing. Both guarded cases above have
a *runtime* check, not a note; that is why they held.

**Corollary worth remembering:** "it works" is not evidence the two agree. It may
mean they are coincidentally compatible today. Prefer a check that would fail if
they diverged over an observation that they currently do not.

Related: [[2026-08-01-2205_chainlit-schema-stale-and-silent-persistence]] is the
same shape across a boundary we do not own — our DDL had to match Chainlit's
expectations, nothing enforced it, and inserts failed silently.
