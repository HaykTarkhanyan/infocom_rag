# Evaluation design

**Provenance:** surveyed 2026-08-01 from `Desktop/metric/Washington`, a
text-to-SQL chatbot project with a mature eval setup. That project answers
questions by generating SQL; we answer them by retrieving documents, so the
*content* of its research is largely inapplicable while its *methodology* and
its mistakes transfer well. File paths below are pointers to go read, surveyed by
a subagent rather than verified line-by-line.

This project currently has **no evaluation of any kind**. Until it does, every
retrieval knob — top-k, chunk size, overlap, hybrid weighting, base vs large — is
guesswork, because retrieval quality has no objective correctness criterion the
way "no content dropped" and "nothing over 512 tokens" do.

---

## 1. Question-set schema

Worth copying nearly as-is from Washington's `tests/fixtures/eval_questions.toml`.
TOML, one table per case, with:

- **`priority = 5..1`** tiers plus a documented run order for when the budget is
  short ("stop if a P5 fails"). Lets a smoke run and a full run share one file.
- **An assertion contract per case**, rather than a single expected answer.

Their SQL assertions were `expected_action_any_of`, `sql_must_contain`,
`sql_must_not_contain`, `answer_must_contain`, `answer_must_not_contain`. The
retrieval equivalent:

```toml
[[question]]
id       = "procurement-conflict-of-interest"
priority = 5
text     = "Ի՞նչ է գրել Ինֆոքոմը շահերի բախման մասին պետական գնումներում"

# retrieval: did we surface the right source at all?
expected_source_ids_any_of = [1164857, 1163360]

# grounding: did the answer use it correctly, and avoid known traps?
answer_must_contain     = ["շահերի բախում"]
answer_must_not_contain = ["2027"]   # the law takes effect later; a common slip
```

Two distinct things are being measured and they should stay separable:
**retrieval recall** (is the right chunk in the top-k at all?) and **grounding**
(given the right chunk, is the answer faithful?). A system can fail either way and
the fixes are completely different.

## 2. Metrics

- Report **bucketed accuracy with Wilson 95% confidence intervals**, not a single
  headline number. With a 30-50 question set, the CI is wide and a 3-point
  "improvement" is usually noise. This is the single most useful idea in their
  evaluation memo.
- Bucket by difficulty and by question type (factual lookup / multi-article
  synthesis / temporal / entity-centric).
- **Hold out 20-25% of cases and never tune against them.** Catches
  eval-overfitting, which is otherwise invisible and inevitable.
- Adopt their three-way per-case verdict: `pass | partial | fail`, where
  *partial* meant "right tables, wrong number". Our analogue: **right chunk
  retrieved, answer subtly wrong** — which is exactly the failure mode most worth
  tracking separately.

## 3. LLM-as-judge

Their recipe, which their own memo recommends over what the codebase actually
does:

- Judge with a **stronger model than the one being graded**.
- **Binary pass/fail**, not a 1-10 score. Scores invite false precision and drift
  between runs.
- Write short **grading notes** per question ("must mention X, must not claim Y")
  rather than a full reference answer. Reference answers over-constrain phrasing
  and are expensive to maintain in Armenian.
- **Cache judgements** keyed on `(question, prediction_hash, judge_model)`.
  Re-running the suite after an unrelated change should be nearly free.

## 4. Runner design

From `evaluation/run_laundris_golden_eval_*.py` and `e2e_ask_eval_*.py`:

- argparse CLI with `--smoke`, `--limit`, `--concurrency`.
- **Always exercise the real entry point** (they POST to the running API) rather
  than importing internal functions. One-off scripts drift from the real code
  path; the eval then passes while production is broken.
- Unique per-run session IDs, so one run's state cannot leak into the next as
  fake history.
- **Dual output**: timestamped JSON for machines, Markdown for humans with the
  retrieved sources and the answer side by side. The Markdown is what actually
  gets read.

## 5. Mistakes not to repeat

Both are visible in Washington's own repo and contradict its own written advice:

1. **Do not hand-roll a heuristic grader.** Their `evaluation/grading.py` is a
   string/number matcher that accreted dozens of one-off patches; the comments
   reportedly name individual failing cases ("r41", "r64", "this slipped
   through"). It became unmaintainable, which is why the memo recommends
   LLM-as-judge instead.
2. **Do not dump loose result files.** ~150 timestamped JSON/CSV/MD files sit in
   `evaluation/` with no compaction, so run-over-run comparison is manual. Build
   the small results store (SQLite or Parquet, one row per case per run) that
   their doc recommends and their code never got around to.

The general lesson: prefer what a mature project's documentation concluded over
what its code accumulated under deadline.

## 6. Design points that transfer to retrieval

- **Hybrid retrieval — BM25 plus dense embeddings — consistently beats either
  alone.** Especially relevant for Armenian proper nouns and institution names,
  where exact lexical match is strong signal and our 1,074-term `infotag`
  taxonomy gives a ready-made entity vocabulary.
- **Retrieved text is untrusted input, never instructions.** Chunks come from a
  public website; delimit them clearly and instruct the synthesis model to treat
  them as data. This is indirect prompt injection and it applies to us even
  though the SQL-specific safeguards do not.
- **Cite sources in every answer.** Their "trust collapse" pitfall: one confident
  wrong answer ends adoption, and citations are what let a reader catch it. We
  already carry `url`, `title`, and `published` on every chunk.
- **Pin model versions and gate upgrades on the eval.** A silent provider-side
  model change otherwise looks like a mysterious quality regression.

## 7. Testing LLM-dependent code

`tests/test_chat_llm_client.py` is a directly reusable pattern: patch the module
client with an `AsyncMock` returning canned JSON, then assert on the *outgoing
payload* rather than the response. Never touches the real API, so it runs in CI
and costs nothing. `tests/async_fakes.py` (a scripted fake that records calls) is
the pattern to copy for a fake vector store.

This matters here because the archived prototype's 71 tests mocked
`torch`/`transformers`/`weaviate` at the `sys.modules` level and consequently
verified nothing that could break.

## 8. What does NOT transfer

Most of Washington's knowledge base is SQL-specific and should not be mined
further: schema linking, DDL prompting, sqlglot AST validation, read-only DB
roles, statement timeouts, row-level security, join-path and NULL-handling
pitfalls, and the Spider/BIRD/EX/VES benchmark literature. There is also **no
Armenian or multilingual handling anywhere in that repo** — it is English-only,
so that part we build from scratch.
