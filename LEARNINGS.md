# Learnings

Non-obvious things discovered while building this, that the code alone does not explain.

---

## Browser tokenizer playgrounds miscount ATE-2 by 53%

**Symptom.** The same Armenian sentence tokenizes to **15 tokens** with Python
`transformers`/`tokenizers`, but **23 tokens** in
[Xenova/the-tokenizer-playground](https://hf.co/spaces/Xenova/the-tokenizer-playground)
using ATE-2's own `tokenizer.json`. (tiktokenizer.vercel.app cannot load ATE-2 at
all -- it only supports a hardcoded list. And its `cl100k_base` gives **141**
tokens for that sentence, because OpenAI's BPE has no Armenian and falls back to
raw UTF-8 bytes at ~2 tokens per character.)

**Root cause.** `transformers.js` does not understand the *newer* serialization
of the SentencePiece `Metaspace` pre-tokenizer, so it silently skips it. Without
Metaspace, no `▁` (U+2581) is prepended to word starts, word-initial forms stop
matching vocab entries -- which are stored *with* `▁` -- and the Unigram model
falls back to shorter fragments.

```
Python : ['<s>', '▁Հայաստանի', '▁Ազգային', '▁ժողով', 'ը', ...]        15 tokens
JS     : ['<s>', 'Հայաստանի',  'Ազգ', 'ային', 'ժ', 'ող', 'ով', 'ը', ...] 23 tokens
```

The JS output decodes back to `ՀայաստանիԱզգային...` -- every space gone. That is
the tell.

**How it was proven.** Two wrong theories died first:

1. *"The `Precompiled` normalizer is being dropped."* Falsified: removing the
   normalizer locally still yields 15 tokens. It does nothing to this text.
2. *"`pre_tokenizer` of type `Sequence` is unsupported."* Falsified: T5 also uses
   a `Sequence` and the playground tokenizes it correctly (8 tokens, matches
   Python).

The actual proof: setting the pre-tokenizer to `WhitespaceSplit()` alone --
i.e. keeping stage 1 and dropping the `Metaspace` stage -- reproduces the
playground's output **ID for ID**:

```python
t = Tokenizer.from_file("tokenizer.json")
t.pre_tokenizer = WhitespaceSplit()          # Metaspace dropped
t.encode(s).ids == playground_ids            # True, all 23
```

**The discriminator.** Same vocab, same architecture, different serialization:

| model | Metaspace serialization | playground |
|---|---|---|
| `intfloat/multilingual-e5-base` (ATE-2's parent) | `add_prefix_space: true` (legacy) | 15 ✅ |
| `xlm-roberta-base` | `add_prefix_space: true` (legacy) | — |
| `Xenova/t5-small` | `add_prefix_space: true` (legacy) | 8 ✅ |
| `Metric-AI/armenian-text-embeddings-1` | `prepend_scheme: "always", split: true` | — |
| `Metric-AI/armenian-text-embeddings-2-*` | `prepend_scheme: "always", split: true` | 23 ❌ |

`multilingual-e5-base` and ATE-2 have **byte-identical vocabularies** and give
15 vs 23 in the same browser. Metric-AI re-serialized the tokenizer with a newer
`tokenizers` version when finetuning; `add_prefix_space` was replaced by
`prepend_scheme` upstream, and the JS port still reads the old field.

**Consequences.**
- Our pipeline is unaffected: `src/chunking.py` uses Python `tokenizers`, which
  is correct. The measured 1.88-1.99 tokens/word for Armenian stands.
- Do **not** use browser playgrounds to reason about ATE-2 chunk budgets. Use
  `research/inspect_tokenizer.py`.
- General lesson: a tokenizer is vocab + merges **+ a normalizer/pre-tokenizer
  pipeline**. The files are portable; the *implementations* are not, and the
  pipeline stages are where reimplementations quietly diverge. It is invisible
  for ASCII English, which is why the bug survives in a 689-like Space.

---

## Armenian tokenizes at ~1.9-2.0 tokens/word through XLM-R

Measured with ATE-2's real tokenizer on our cleaned corpus, not estimated. So the
512-token model cap is roughly **257 words**.

Our 94 research/data-driven articles have a median of **3,001 tokens** and
**90 of 94 (96%) exceed 512**. This is the concrete reason the archived prototype
was unusable: it embedded whole articles with `truncation=True`, so almost the
entire corpus was silently cut from its own vectors while the full text was still
sent to the LLM.

Both ATE-2 variants share one `tokenizer.json` (verified by sha256), so chunk
boundaries do not change between base and large -- only the embedding dimension
does (768 vs 1024).

---

## selectolax needs `separator="", strip=False`

For extracting text from WordPress `content.rendered`:

- `strip=True` strips whitespace from each text node, gluing words across
  `<span>` boundaries: `բախում։Ըստ`.
- `separator=" "` inserts a space at every inline boundary, detaching Armenian
  punctuation: `օրենքի ՝` instead of `օրենքի՝`.
- `separator="", strip=False` preserves the HTML's own whitespace, which is
  authoritative. Collapse runs of whitespace afterwards.

Either mistake corrupts tokenization on every chunk.

---

## Compare extractor outputs only after normalizing whitespace

A raw set-diff of two extractors' output lines suggested selectolax recovered 12%
more content than trafilatura. After normalizing whitespace the difference
vanished entirely -- it was segmentation artifacts. The extractors agree on
content; they differ on spacing and heading handling.

---

## Verify the verifier

A coverage check reported that chunking had lost 117 sentences. The chunker was
fine; the *check* was wrong -- it concatenated chunks including their
`passage: title / heading` prefixes, which interrupts the source text. Re-run at
paragraph level against chunk bodies only: 4,318 of 4,318 preserved.

Before believing a regression, confirm the test itself is sound.

---

## Windows / environment gotchas

- `huggingface_hub` symlinks blobs into snapshots and dies with
  `OSError: [WinError 1314] A required privilege is not held by the client`
  without admin or Developer Mode. Fix: `HF_HUB_DISABLE_SYMLINKS=1`.
- `HF_HOME` must be in `os.environ` **before** importing `huggingface_hub` or
  `transformers` -- it is read at import time, so `load_dotenv()` has to run
  first.
- `PYTHONIOENCODING=utf-8` is required for any script that prints Armenian;
  otherwise Windows `cp1252` raises `UnicodeEncodeError`.
- WordPress `X-WP-TotalPages` reflects the `per_page` of *that* request. Probing
  with `per_page=1` and then fetching with `per_page=100` walks off the end into
  HTTP 400. Compute pages from `X-WP-Total` and your own page size.
