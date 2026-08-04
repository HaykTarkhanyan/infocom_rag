# An extractor tuned on one corpus fails silently on the next

`extract_content` was written against the 94 `indepth` articles and worked
perfectly on them: 0 empty, 0 suspiciously short, 4,318 of 4,318 source
paragraphs preserved. It was then pointed at `news` and started returning
**empty strings** for some posts. No exception, no warning — a document with no
text.

**Cause.** It collects text from block elements only:

```python
BLOCK_SELECTOR = "p, h1, h2, h3, h4, h5, li, blockquote, figcaption, td, th"
```

No `div`. `indepth` articles are Elementor-built and always contain `<p>`, so
every one of them matched. Some `news` posts carry their whole body as bare text
inside `<div>`s and match **nothing**.

Measured on one month: **7 of 1,236 (0.6%)**, each holding 43-207 real words.
Projected across the year that is ~116 documents that would have been embedded,
indexed and retrieved as empty.

## Why the obvious fix is wrong

Adding `div` to `BLOCK_SELECTOR` looks like a one-word fix and is not. **Divs
nest.** Every ancestor div would re-emit all of its descendants' text, and the
existing de-duplication only collapses *consecutive* identical blocks:

```html
<div><div><div>Եզակի նախադասություն։</div></div></div>
```

would yield the sentence three times. The fix is a **fallback**: if the
structured pass produced nothing, take the whole tree's text once. The normal
path is untouched — verified at 0 changes across 400 previously-good records —
and the fallback is logged at WARNING so a second extraction strategy firing is
visible in the fetch log rather than silent.

Result: the full-year fetch produced **1** empty text instead of ~116.

## The rule

**A parser's test set is the corpus it was written against.** Its silence on a
new corpus is not evidence it works there — an extractor's failure mode is
usually *less output*, which looks exactly like a short document.

When pointing existing extraction at a new source:

1. **Count the empties and the suspiciously-shorts before trusting the run.** One
   query over the output (`text == ""`, `n_words < 60`) would have caught this in
   seconds, and did.
2. **Compare against a naive baseline.** `HTMLParser(html).text()` on the same
   input immediately showed 342-2,051 characters where our extractor returned 0.
   The gap between the careful extractor and the dumb one is the bug.
3. **Keep the raw source.** `content_html` is stored on every record precisely so
   an extraction bug found later costs a re-parse, not a 175 MB re-download.

Related: [[2026-08-02-1520_a-metric-nobody-prints-rots]] — same family. There a
metric was computed and never read; here output was produced and never counted.
Both are failures of not looking, not of logic.
