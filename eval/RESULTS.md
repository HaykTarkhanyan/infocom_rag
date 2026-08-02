# Eval results — first run, 2026-08-02

Corpus: 94 articles / 969 chunks. Retriever: dense (ATE-2-large). Generator:
`openai/gpt-5.4-mini` at temperature 0.

## Retrieval — automated, reproducible

`python eval/run_eval.py --no-judge`

```
recall@k  27/29 = 93.1%   95% CI [78.0%, 98.1%]
MRR       0.835
rank of first correct: median 1, worst 7
```

Both apparent misses were investigated, and **neither is a retrieval failure**:

- `q-627756-spec` — retrieved a *different* article (53468) that answers the
  question just as well. The ground truth listed only the article the question
  was generated from. Fixed by adding 53468 to `expected_post_ids`. Lesson:
  single-source ground truth undercounts recall whenever the corpus covers a
  topic more than once.
- `3-806341-spec` — the correct article ranked **#2** at distance 0.5249, just
  outside the 0.50 threshold. A threshold false-negative, which is what prompted
  the sweep below.

Corrected recall is **28/29 (96.6%)**.

## Threshold, tuned properly

Swept offline against the vectors (free — no LLM calls). Tuned on non-holdout
questions **only**, then verified against the held-out set:

| max_distance | recall@10 (tuned, n=23) | recall@10 (held out, n=6) |
|---|---|---|
| 0.40 | 18/23 | — |
| 0.45 | 20/23 | — |
| 0.50 | 21/23 | 6/6 |
| **0.55** | **22/23** | **6/6** |
| 0.60 | 22/23 | — |

0.55 adopted. Held-out accuracy did not drop, so the choice generalises rather
than fitting the eval.

The cost of loosening is that fewer unanswerable questions get declined at the
*retrieval* stage (2/6 at 0.50, 0/6 at 0.55). That is **not** a correctness cost
here — the model declined correctly on all six either way — only a saved LLM
call. It would become a real safety net with a weaker generation model.

## Grounding — manual, NOT reproducible

**Graded by Claude (Opus 5) in-session, at the user's suggestion, because the
pinned judge costs ~$3.28 per run.** Two caveats that matter:

1. **Not independent.** The same assistant chose the model, wrote the system
   prompt, generated the questions and set the threshold. This is the least
   impartial judge available and it grades its own design decisions.
2. **Not reproducible.** An eval that needs a human-in-the-loop assistant cannot
   be re-run on demand, pinned to a version, or gated in CI.

`eval/run_eval.py` therefore keeps `openai/gpt-5.2-pro` as the automated judge —
the only model clearly stronger at Armenian reading (0.973) than the one being
graded (0.965). Judgements cache by `(question, answer, judge_model)`, so the
cost is paid once. Treat the results below as a first signal, not a measurement.

### Unanswerable and partial-coverage cases: 6/6 handled correctly

This is the axis that matters most — it is the only one that detects
hallucination.

| id | behaviour | verdict |
|---|---|---|
| `unans-football` | declined at retrieval, no LLM call | pass |
| `unans-gyumri-population` | declined at retrieval | pass |
| `unans-yerevan-sevan-road` *(trap, holdout)* | retrieved road/transport chunks, still declined, explained what the excerpts *do* cover | **pass** |
| `unans-gas-price` | declined on the price, correctly gave the import volume that IS in the corpus | pass |
| `unans-future-court` | declined, and reasoned that the newest relevant excerpt is dated 2026-01-13 so cannot cover August | **pass** |
| `unans-state-debt` | gave the $14.5bn figure **with its real date** and said the asked date is not covered | pass |

The last one exposed a **bug in the eval set, not the system**. The corpus does
contain Armenian state debt as of 2026-03-31, verified in the source text:
`2026թ մարտի 31-ի դրությամբ պետական պարտքը 14․5 մլրդ ԱՄՆ դոլար է`. The question
was mislabelled `unanswerable` and its grading note ("FAIL if it states any debt
figure") would have failed a correct answer. Reclassified as
`partial_coverage` with corrected notes.

The two hardest cases — the topically-adjacent road trap and the after-cutoff
court question — were handled well, which is the behaviour the system prompt asks
for and the failure mode most likely to destroy trust.

### Factual answers: sample of 5, all sound

All retrieved at rank 1. Spot-checked claims traced to the excerpts: exact
figures preserved (`4-5 օրում, 1 շաբաթում`), speakers correctly attributed
(Իվետա Տոնոյանի, Արամ Մինասյանի), and reasoning stayed inside the sources.

## Re-scored after the question repair — 2026-08-02

Scored offline against `vectors_large.npz` (no LLM calls), at the current
`max_distance = 0.55`, over the repaired set:

```
recall@10  30/30 = 100.0%
MRR        0.938
coverage   95.0%
```

Up from 27/29 = 93.1%. Three separable causes, and only one is real improvement:

1. **The threshold fix.** `3-806341-spec` sat at distance 0.5249 and was cut by
   the old 0.50; it now returns at rank 2.
2. **`unans-state-debt` became scorable** once its `expected_post_ids` was
   restored, and hits at rank 1 (n went 29 → 30).
3. **Two rewritten questions retrieve better**: `q-806535-broa` moved rank 3 → 1
   and `4-44-625820-broa` rank 4 → 1. The other six were unchanged at rank 1.

So the repair did not cost recall — it helped, because naming a question's
subject gives the retriever more to match. Which is the same property that makes
the questions leakier, and is why overlap is now tracked per question.

## The set can no longer tell retrievers apart

This is the number that matters most, and it is bad news for the eval rather
than for the system:

| | R@1 | R@3 | R@10 | MRR |
|---|---|---|---|---|
| dense | 27/30 = 90.0% | 30/30 = 100% | **30/30 = 100%** | 0.944 |
| bm25 | 26/30 = 86.7% | 28/30 = 93.3% | **30/30 = 100%** | 0.908 |

Every Wilson interval overlaps. At `top_k = 10` — the configured operating point
— they are **tied**. Detecting a 5-point gap at 80% power would take roughly
**685 questions per arm**; there are 30.

The cause is corpus size: picking the right article out of **94** is easy for
any retriever. The set should regain its discriminative power on the full 5,836
article corpus, and that is the run that would actually settle dense vs BM25 —
see the revisit note on DECISIONS.md #18.

**Do not read 100% as "retrieval is solved."** Read it as "this eval has stopped
measuring retrieval at this corpus size."

## What recall@k does not tell you (added 2026-08-02)

A review of the harness found two metrics being computed and silently discarded.
Both are now printed. Recomputed over the same run:

**Multi-source coverage.** `recall@k` scores a question as a hit if *any*
expected article is retrieved, so the two 3-source questions were both counted as
clean hits:

| id | reported | articles actually found |
|---|---|---|
| `kkh-multi` | hit @ rank 1 | 2 of 3 |
| `kgmsn-multi` | hit @ rank 7 | **1 of 3** |

Mean coverage is **89.7%** against the headline 93.1% recall. With only 2
multi-source questions this is a signal, not a measurement — but the direction is
clear and the metric was invisible before.

**Assertions: 17/27 = 63% passed.** These are literal substring checks and were
never printed by the report. Reading the ten failures, most are **artefacts of
Armenian morphology, not wrong answers**:

| assertion | what the answer said | real? |
|---|---|---|
| `Կառավարությունը` | `Կառավարությունում`, `Կառավարությունից` | no — case inflection |
| `արդարացի չէ` | `արդար չէ`, `ճիշտ չէ` | no — synonym |
| `87%` | `ճնշող մեծամասնությունը` | **yes** — the figure was dropped |

So 63% is a floor, not an accuracy. The check earns its keep as a *pointer* to
answers worth reading, and the report now says so in the output itself.

## Known gaps

- Grounding across the 27 factual questions is **sampled, not measured**. Run the
  pinned judge for a real number.
- 35 questions gives wide confidence intervals — the 93.1% recall CI spans
  78–98%. Treat single-point differences below ~10 points as noise.
- Questions were LLM-drafted from known chunks, then filtered: 3 dropped for
  hallucinated `must_contain` strings, 12 for >0.65 token overlap with their
  source. Median overlap after the 2026-08-02 repair is **0.46**, max 0.65, so
  the set still leans easier than real user questions.
- Only 2 multi-source questions, and none tests conflicting sources.
- **No multi-turn cases.** Query rewriting shipped 2026-08-02 (DECISIONS #21) and
  the eval does not exercise it at all — `run_eval.py` sends no `history`, which
  keeps these numbers comparable with earlier runs but leaves the whole
  conversational path unmeasured. See DEFERRED_TODO.md for what that bucket needs.
- `results.jsonl` holds two runs, both at `max_distance = 0.50` with no judge.
  The 0.55 threshold now in `config.toml` was chosen from an offline sweep and
  has **never been through a full recorded run**.
