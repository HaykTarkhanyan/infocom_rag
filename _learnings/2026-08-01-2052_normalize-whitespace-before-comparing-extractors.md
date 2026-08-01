# Compare extractor outputs only after normalizing whitespace

**Symptom.** A raw set-diff of output lines suggested selectolax recovered ~12%
more content than trafilatura (14,957 vs 13,323 words over 12 articles), which
would have been a strong reason to pick it.

**Cause.** The difference was almost entirely whitespace and segmentation, not
content. The same sentence appeared in both, spelled differently:

```
selectolax : Ըստ Հանրային ծառայության մասին օրենքի ՝ շահերի
trafilatura: Ըստ Հանրային ծառայության մասին օրենքի՝ շահերի
```

Set membership treats those as two distinct lines. Sentence-level diffs were no
better -- one extractor marks headings `## `, the other inlines the byline into
the following paragraph, so sentence boundaries land in different places.

**Fix.** Normalize before comparing: collapse whitespace, re-glue punctuation,
strip heading markers. After that the extractors agreed on content and the choice
came down to speed (selectolax ~12x faster) and heading preservation.

**Consequence.** Do not let a raw diff pick your extractor. Normalize first, or
you will optimize for formatting noise. The same applies to any "did I lose
content?" comparison between two text pipelines.
