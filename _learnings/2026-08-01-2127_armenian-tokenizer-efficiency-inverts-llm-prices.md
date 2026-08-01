# Armenian tokenizer efficiency inverts LLM price comparisons

**Symptom.** `openai/gpt-5.4-mini` lists at $0.75/$4.50 per 1M tokens and
`google/gemini-3-flash-preview` at $0.50/$3.00, so Gemini looks 33% cheaper. It
is not — for Armenian it is *more* expensive.

**Measured**, 2026-08-01, on five real chunks from our own corpus (784 words of
Armenian) sent to both models through OpenRouter, comparing the `prompt_tokens`
each reported back:

| model | prompt tokens | tokens/word | measured cost |
|---|---|---|---|
| openai/gpt-5.4-mini | 1,952 | **2.49** | **$0.001486** |
| google/gemini-3-flash-preview | 3,189 | **4.07** | $0.001597 |

Gemini needed **63% more tokens for identical text**, which more than cancels its
lower headline rate.

**Cause.** Tokenizer vocabulary coverage of the Armenian script differs sharply
between model families. A tokenizer with poor Armenian coverage falls back to
shorter subwords or bytes, inflating the token count for the same characters.
Nothing about the price page reveals this.

**Consequences.**

- **Compare USD per Armenian *word*, not per token.** The headline rate is only
  half the calculation; the other half is the tokenizer, and it can be a 1.6x
  factor.
- This is measurable in one cheap API call per candidate: send identical text,
  read `usage.prompt_tokens` from the response. No tokenizer download and no
  guessing — the provider tells you.
- It also explains the ArmBench-LLM spend report, where gpt-5.4-mini cost $1.82
  to run the full benchmark against gemini-3-flash-preview's $3.28 despite the
  higher unit price. The benchmark blog names tokenizer efficiency as a hidden
  cost factor for Armenian; this is that effect, quantified on our data.
- Cuts the other way too: context *windows* are counted in tokens, so a poor
  Armenian tokenizer also shrinks how much Armenian actually fits.

**Related.** Our embedding model measures 1.99 tokens/word on the same corpus
(XLM-R, 250k vocab) — better than either generation model, because it was
finetuned for Armenian. See
[[2026-08-01-2052_armenian-token-ratio-and-512-cap]].

**Caveat.** Measured on one 784-word sample of news prose. The ratio will move
with content type (numbers, Latin-script names and quoted English all tokenize
differently). Re-measure before relying on it for a materially different corpus.
