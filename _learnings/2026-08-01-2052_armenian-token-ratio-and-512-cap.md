# Armenian runs ~1.9-2.0 tokens/word through XLM-R, so 512 tokens is ~257 words

Measured with ATE-2's real tokenizer on our cleaned corpus, not estimated.

**Numbers.** On the 94 research + data-driven articles:

```
tokens/word : 1.99 mean  (1.88 on a clean single sentence)
median article : 3,001 tokens
max article    : 18,400 tokens
total          : 344,389 tokens
over 512       : 90 of 94 articles (96%)
```

**Consequences.**

This is the concrete reason the archived prototype was unusable. It embedded
whole articles with `truncation=True`, so 96% of this corpus was silently cut
from its own vectors while the *full* text was still sent to the LLM. Retrieval
was blind to most of what it later quoted, with no error and no warning.

Chunking must respect the cap by splitting, never by truncating.

**Both ATE-2 variants share one tokenizer** (verified by sha256 of
`tokenizer.json`: `3a56def25aa40fac` for base and large alike). So chunk
boundaries do not change between base and large -- only the embedding dimension
does, 768 vs 1024. `data/chunks.jsonl` stays valid across that switch; only the
vector store needs rebuilding.

The 512 figure is architectural (`max_position_embeddings=514`), not a tunable.

Beware: an earlier estimate of 2.2-2.8 tokens/word came from measuring ATE-1's
tokenizer on *raw scraped* text that still contained URLs and markup noise. Clean
text tokenizes more efficiently. Measure on the text you will actually embed.
