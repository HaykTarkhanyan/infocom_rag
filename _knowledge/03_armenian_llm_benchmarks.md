# Armenian LLM benchmarks — ArmBench-LLM 1.0

**Sources.** Blog published 2026-04-02, leaderboard snapshot captured 2026-06-15,
both saved verbatim in `Desktop/metric/ArmBench-LLM/references/`. Live leaderboard:
<https://metric-ai-armbench-llm.hf.space/>. Dataset:
<https://huggingface.co/datasets/Metric-AI/ArmBench-LLM-data>. By Metric AI Lab —
the same lab that publishes the ATE-2 embeddings we use, so the benchmark and our
embedding model share a maintainer.

26 models, evaluated over OpenRouter (open models under 30B were run locally).
Task groups: NER, POS, Reading Comprehension, Classification, MCQA, Generation,
Translation, Armenian Unified Exams, Text Processing, MMLU-Pro-Hy.

---

## 1. Read the right column, not the Average

The headline ranking is by `Average`, and for our purposes it is close to useless.
The snapshot says so itself:

> Scale note: columns use different metrics and scales. Most task columns are 0-1
> (accuracy / F1 / etc.), but Exams are point sums and some Generation/Translation
> sub-metrics can exceed 1. Compare within a column, not across.

Exams run to ~18 points, MS MARCO to ~34, Belebele is 0-1. An "Average" over
those is dominated by whichever column has the widest range, and it rewards
Armenian *knowledge* (history, literature, exams) that our system does not need —
we hand the model the source text.

**For RAG, the relevant column is Reading Comprehension**, and secondarily
Generation. Reading is what our synthesis step actually does: answer a question
from provided passages.

## 2. Reading comprehension, top models

Reading aggregates SQuAD, Belebele, DREAM, Hartak (Armenian public-services MCQA)
and MS MARCO. The three clean 0-1 sub-scores are the most interpretable:

| model | Reading | Belebele | DREAM | Hartak | bench spend |
|---|---|---|---|---|---|
| openai/gpt-5.4-mini | **0.6228** | 0.96 | 0.98 | 0.9556 | $1.82 |
| openai/gpt-5.2-pro | 0.6096 | 0.96 | 0.98 | 0.9778 | $160.20 |
| qwen/qwen3.5-27b | 0.6065 | 0.92 | 0.94 | 0.8222 | (local) |
| x-ai/grok-4-fast | 0.6037 | 0.82 | 0.92 | 0.9556 | $2.71 |
| anthropic/claude-3.7-sonnet | 0.5966 | 0.86 | 1.00 | 0.9111 | $16.49 |
| google/gemini-3-pro-preview | 0.5901 | 0.90 | 0.94 | 0.8889 | $67.26 |
| google/gemini-3.1-pro-preview | 0.5412 | 0.88 | 0.88 | 0.7556 | $44.60 |
| **google/gemini-3-flash-preview** | **0.5254** | 0.74 | 0.94 | 0.80 | **$3.28** |
| google/gemini-2.5-flash | **0.4895** | 0.76 | 0.92 | **0.4444** | $4.55 |

## 3. What this changes for us

**Our configured default was the wrong Gemini.** `gemini-2.5-flash` has the
weakest Reading score in the top ten (0.4895) and a Hartak score of 0.4444 — by
far the worst of any capable model, on the one sub-task closest to "answer an
Armenian question from Armenian source material". It also cost *more* to run the
full benchmark than `gemini-3-flash-preview` did.

Switched the default to **`google/gemini-3-flash-preview`**: better Reading
(0.5254 vs 0.4895), far better Hartak (0.80 vs 0.4444), ranked #1 overall, and
the blog's "best overall value" pick. It is also what the Washington project
independently settled on.

**Cost nuance worth keeping straight.** The benchmark spend report is *total
suite cost*, which conflates three things the blog names explicitly: unit price,
**tokenizer efficiency on Armenian script**, and reasoning verbosity. At
OpenRouter list prices, `gemini-3-flash-preview` ($0.50/$3.00 per M) is
*more* expensive per token than `gemini-2.5-flash` ($0.30/$2.50). For our
workload (~5k input, ~500 output per question) that is roughly $0.0040 vs
$0.0028 per query. It ran the whole benchmark cheaper because it used fewer
tokens overall, not because its rate is lower. Both facts are true; do not quote
the spend report as a per-token price.

## 4. Honest note: Gemini is not the best choice for this task

You asked for Gemini and that is what is configured. But on the column that
matters for RAG, **`openai/gpt-5.4-mini` beats every Gemini model** — Reading
0.6228 vs 0.5254 for the best Gemini — while being the cheapest capable model in
the spend report ($1.82). `x-ai/grok-4-fast` is also ahead of every Gemini on
Reading at $2.71.

The gap is not small: on Hartak, gpt-5.4-mini scores 0.9556 against
gemini-3-flash-preview's 0.80. If answer quality on Armenian source material
turns out to be the bottleneck, that is the first thing to try — and since model
is one line in `config.toml` and every call is cost-logged, it is a cheap
experiment once the eval set exists.

Gemini keeps two genuine advantages: it is the **gold standard for Armenian
knowledge** (history, literature, exams), which matters if we ever answer beyond
the retrieved context, and Gemini 3 Flash is the best model tested for
**translation**, which matters for cross-lingual questions.

## 5. Caveats that affect how much to trust these numbers

1. **A near-zero score usually means "failed the output format", not "cannot do
   the task."** The 0.1 framework in that repo extracts labels with regex
   (`label_extraction_patterns.py`), and several results look like harness
   failures rather than capability: `gemini-2.5-pro` scores 0.1802 on MMLU-Pro
   while `gemini-2.5-flash` scores 0.6416, and 0.0006 on Simple QA;
   `gemini-3.1-pro-preview` scores 0.0340 on FiNER NER and 0.2278 on Text
   Processing while otherwise ranking mid-table. Treat isolated near-zeros as
   suspect, not as evidence.
2. **Uniform simple prompts.** The blog states no model-specific prompt tuning
   was done, which it acknowledges penalises smaller models most. Scores are a
   floor, not a ceiling.
3. **Excluded models.** Claude 4.5 and 4.6 were dropped for reliability problems
   during evaluation; Grok 4.20 misbehaved via API but reportedly not via its UI.
   Absence from the leaderboard is not evidence of weakness.
4. **The newest Gemini models are not benchmarked at all.** The leaderboard
   covers 2.5-flash, 2.5-pro, 3-flash-preview, 3-pro-preview and
   3.1-pro-preview. OpenRouter currently also offers `gemini-3.5-flash`,
   `gemini-3.6-flash`, `gemini-3.1-flash-lite` and `gemini-2.5-flash-lite`, none
   of which appear here. Our fallback is currently `gemini-3.1-flash-lite`, which
   is therefore **unmeasured on Armenian** — a known unknown.
5. **Benchmark ≠ our task.** None of these tasks is "answer a question from ten
   retrieved Armenian news chunks, with citations, without hallucinating."
   Reading Comprehension is the closest proxy available, and a proxy is all it
   is. This is an argument for building our own eval set (see
   [02_evaluation_design.md](02_evaluation_design.md)), not for trusting the
   leaderboard as a verdict.

## 6. Other findings worth remembering

- **Better globally ≠ better for Armenian.** Gemini 3 *Flash* outperforms Gemini
  3 *Pro* on Armenian despite the reverse being true on global leaderboards. Do
  not pick an Armenian model from general rankings.
- **Open models are viable now.** `qwen/qwen3.5-27b` (Reading 0.6065) beats
  600B+ models like GLM-5 and Mistral-Large and runs on a single GPU. Relevant if
  this ever needs to run on-prem or without an API budget.
- **Tokenizer efficiency is a hidden cost multiplier for Armenian.** Same idea we
  measured for embeddings: Armenian runs ~1.99 tokens/word through XLM-R. Model
  families differ here, so two models at the same headline price can differ
  materially in real cost.
- The blog recommends `gemini-2.5-flash` specifically for **summarization and
  generation** despite its weak Reading score. Our task sits between the two —
  reading-grounded generation — which is another reason to measure on our own
  data rather than pick from either recommendation.
