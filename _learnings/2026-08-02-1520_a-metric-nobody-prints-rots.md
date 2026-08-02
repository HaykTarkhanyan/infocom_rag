# A metric nobody prints will rot, and nobody will notice

Reviewing `eval/run_eval.py` turned up five defects. Not one was a wrong formula
— the Wilson interval is correct, `score_retrieval` does exactly what it says.
Every single one was **plumbing**: a value computed and not printed, a slice
taking the wrong rows, a cache key missing a field.

| defect | how it stayed hidden |
|---|---|
| `report()` `KeyError` on an API-error row | no run had ever errored |
| assertions computed, never printed | output went only to `results.jsonl` |
| `coverage` computed, never printed | same |
| `--limit N` = `questions[:N]` = all holdout | nobody checked *which* 5 ran |
| judge cache key omitted the excerpts | the judge had never been run |

## The compounding failure

The assertion check had a ~50% false-alarm rate — Armenian is agglutinative and
these are literal substring matches, so an answer saying `Կառավարությունից`
fails an assertion demanding `Կառավարությունը`, and `արդար չէ` fails
`արդարացի չէ`.

That decay was *possible* precisely because the output was invisible. Nobody
looked, so nothing pushed back. **An unread metric is not neutral, it is
rotting** — it keeps accumulating a number that no longer means what its name
says. This is the same shape as yesterday's `NameError` in the judge cache,
which survived review because every run used `--no-judge`: the unexercised path
is where the bugs live, and "unexercised" includes "computed but never read".

## A theory I tested and was wrong about

Seeing two hand-rewritten questions jump past the 0.65 overlap threshold, I
guessed the overlap metric was really measuring question *length* — it counts
stopwords, so a longer question should score higher mechanically.

Measured it:

```
correlation(raw overlap, question length) = -0.10
median raw overlap     0.53
median content overlap 0.47   (excluding 38 tokens appearing in >30% of chunks)
```

Essentially no correlation, and content-word overlap tracks raw overlap closely.
**The metric was fine; my rewrites really did leak.** Rephrasing to name the
subject in fewer words took one from 0.77 back to 0.36.

Worth keeping because the wrong version is the comfortable one: it would have
let me dismiss an inconvenient number as an artefact instead of fixing the
questions.

## The rules

1. **If a metric is worth computing, print it.** Storing it for later analysis
   means never.
2. **Print the caveat next to the number**, in the tool's own output, not just
   in a README. `run_eval.py` now tells the reader that assertion failures are
   morphology-prone and need reading before they are believed. The number cannot
   defend itself, and the person reading it in six months is you.
3. **Before dismissing a number as an artefact, measure the artefact.**

Related: [[2026-08-02-1214_two-sources-that-must-agree]] — same family, in that
both are failures of something that was never checked rather than something
computed wrongly.
