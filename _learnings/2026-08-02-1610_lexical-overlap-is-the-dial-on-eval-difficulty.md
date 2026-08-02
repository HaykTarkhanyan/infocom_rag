# Question/source lexical overlap decides what a retrieval eval can measure

The eval set scored **100% recall@10 for both dense and BM25** — it had stopped
measuring retrieval. Three attempts to fix it, with the numbers.

## Attempt 1: generate against a distractor — FAILED

Theory: questions were easy because each was drafted from one chunk in isolation,
so the generator never had to write something *discriminating*. Fix: find the
most confusable article pairs (the corpus has 21 of 94 articles with a >0.85
neighbour; the closest pair is 0.948) and ask for a question answerable from A
but not B.

| set | median overlap | dense@1 | bm25@1 | retrievers disagree |
|---|---|---|---|---|
| existing questions.toml | 0.46 | 27/30 | 26/30 | 5/30 |
| **v1, distractor only** | 0.53 | 15/16 | 15/16 | **0/16** |

Worse than what it replaced. The prompt asked for paraphrase and the model
ignored it — overlap went *up*. Margins did shrink (median +0.123 → ~+0.08, some
near zero), so the questions genuinely sat closer to their distractor. It did not
matter: rank-1 is a binary and a +0.01 margin still passes it.

## Attempt 2: mechanically ban the article's distinctive vocabulary — WORKED

Compute each article's high-TF-IDF terms and forbid them in the prompt.

| set | median overlap | dense@1 | bm25@1 | disagree |
|---|---|---|---|---|
| **v2, distractor + banned terms** | 0.44 | 12/16 | 12/16 | **5/16** |

**Asking a model to paraphrase does not work. Removing the words does.**

## The finding worth keeping

Pooling all 32 hard candidates and bucketing by question/source overlap:

| overlap | dense@1 | bm25@1 |
|---|---|---|
| **< 0.40** | 7/8 | **5/8** |
| 0.40 – 0.55 | 9/11 | 9/11 |
| **> 0.55** | 11/13 | **13/13** |

High-overlap questions are a word-matching problem and BM25 goes 13/13.
Low-overlap questions have no words to match and dense wins. The original set's
median of 0.46 sits exactly in the crossover, which is why it scored the two
retrievers identically.

**The set was not merely too easy — it was concentrated in the one band where the
retriever choice does not matter.** An eval meant to compare retrievers needs a
*spread* of overlap and should report results per band, not one pooled number.

(n is small: 7/8 vs 5/8 is not significant. The direction matches theory and is
the first evidence for hybrid retrieval — the two retrievers win *different*
questions, not the same ones by different margins.)

## Second-order effect: the screen removes hardness along with defects

Mechanically screening the 24 diverse candidates (reject bare deictics, reject
`must_contain` strings absent from the article, reject overlap > 0.55, reject
unretrievable) left 9 — a 38% yield.

But retriever disagreement among survivors was **1/9**, against **6/24** in the
raw batch. The screen threw out the interesting ones: the single unretrievable
candidate and several hard-but-deictic ones.

Banning distinctive terms also *degraded* `must_contain` (5 of 15 drops), because
the model was steered away from exactly the vocabulary those assertions need.
Fixing the question made the assertion worse. Next iteration should generate
`must_contain` in a separate pass that CAN see the banned words.

## Rules

1. **Overlap is the difficulty dial.** Measure it per question, and report
   retrieval results bucketed by it.
2. **To force paraphrase, delete the vocabulary — do not request it.**
3. **Watch what your quality filter removes.** A screen tuned on defects will
   quietly remove difficulty too, because hard questions and broken questions
   look alike from the outside.

Evidence files: `eval/hard_candidates_v1_noban.jsonl` (the failure),
`eval/hard_candidates_v2_clustered.jsonl` (worked but topically clustered),
`eval/hard_candidates.jsonl` (12 disjoint pairs),
`eval/hard_candidates_screened.jsonl` (the 9 survivors).

Related: [[2026-08-02-1520_a-metric-nobody-prints-rots]].
