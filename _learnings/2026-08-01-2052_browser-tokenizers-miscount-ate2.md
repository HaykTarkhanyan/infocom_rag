# Browser tokenizer playgrounds miscount ATE-2 by 53%

**Symptom.** The same Armenian sentence tokenizes to **15 tokens** with Python
`transformers`/`tokenizers`, but **23 tokens** in
[Xenova/the-tokenizer-playground](https://hf.co/spaces/Xenova/the-tokenizer-playground)
loading ATE-2's own `tokenizer.json`.

(Separately: tiktokenizer.vercel.app cannot load ATE-2 at all -- it only supports
a hardcoded list of 53 models. Its `cl100k_base` gives **141** tokens for that
sentence, because OpenAI's BPE has no Armenian and falls back to raw UTF-8 bytes
at ~2 tokens per character.)

**Cause.** `transformers.js` does not understand the *newer* serialization of the
SentencePiece `Metaspace` pre-tokenizer, so it silently skips that stage. Without
Metaspace, no `▁` (U+2581) is prepended to word starts; word-initial forms stop
matching vocab entries -- which are stored *with* `▁` -- and the Unigram model
falls back to shorter fragments.

```
Python : ['<s>', '▁Հայաստանի', '▁Ազգային', '▁ժողով', 'ը', ...]            15 tokens
JS     : ['<s>', 'Հայաստանի', 'Ազգ', 'ային', 'ժ', 'ող', 'ով', 'ը', ...]   23 tokens
```

The JS output decodes back to `ՀայաստանիԱզգային...` -- every space gone. That is
the tell.

## Two theories tested and disproven first

1. **"The `Precompiled` normalizer is being dropped."** Falsified: removing the
   normalizer locally still yields 15 tokens. It does nothing to this text.
2. **"`pre_tokenizer` of type `Sequence` is unsupported."** Falsified: T5 also
   uses a `Sequence` and the playground tokenizes it correctly (8 tokens,
   matching Python).

## Proof

Keeping stage 1 and dropping only the `Metaspace` stage reproduces the
playground's output **ID for ID**:

```python
t = Tokenizer.from_file("tokenizer.json")
t.pre_tokenizer = WhitespaceSplit()          # Metaspace dropped
t.encode(s).ids == playground_ids            # True, all 23
```

## The discriminator: same vocab, different serialization

| model | Metaspace serialization | playground |
|---|---|---|
| `intfloat/multilingual-e5-base` (ATE-2's parent) | `add_prefix_space: true` (legacy) | 15 ✅ |
| `Xenova/t5-small` | `add_prefix_space: true` (legacy) | 8 ✅ |
| `xlm-roberta-base` | `add_prefix_space: true` (legacy) | — |
| `Metric-AI/armenian-text-embeddings-1` | `prepend_scheme: "always", split: true` | — |
| `Metric-AI/armenian-text-embeddings-2-*` | `prepend_scheme: "always", split: true` | 23 ❌ |

`multilingual-e5-base` and ATE-2 have **byte-identical vocabularies** and give 15
vs 23 in the same browser. Upstream `tokenizers` replaced `add_prefix_space` with
`prepend_scheme`; Metric-AI re-serialized with a newer version when finetuning,
and the JS port still reads the old field. An unknown key is indistinguishable
from an absent one, so it defaults to "off" with no error.

**Consequences.**
- Our pipeline is unaffected: `src/chunking.py` uses Python `tokenizers`.
- Do **not** use browser playgrounds to reason about ATE-2 chunk budgets. Use
  `research/inspect_tokenizer.py`.
- General: a tokenizer is vocab + merges **+ a normalizer/pre-tokenizer
  pipeline**. The files are portable; the *implementations* are not, and the
  pipeline stages are where reimplementations quietly diverge. It is invisible
  for ASCII English, which is why the bug survives in a 689-like Space.
