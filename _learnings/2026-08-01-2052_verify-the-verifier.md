# Verify the verifier before believing a regression

**Symptom.** A coverage check reported that chunking had silently dropped **117
of 6,780 sentences** (1.7%) -- exactly the failure mode the rewrite existed to
prevent, so it looked like the new chunker had reproduced the old bug.

**Cause.** The chunker was fine. The *check* was wrong. It rebuilt each article
by concatenating its chunks, but each chunk begins with a
`passage: {title}\n{heading}\n\n` context header. Concatenating whole chunks
therefore splices that header into the middle of the reconstructed text, so any
sentence spanning a chunk boundary fails a substring test:

```
...end of chunk N   passage: Title Heading   start of chunk N+1...
                    ^^^^^^^^^^^^^^^^^^^^^^ interrupts the source text
```

**Fix.** Compare against chunk *bodies* only, at paragraph granularity:

```
PARAGRAPH-level coverage: 4318/4318 preserved (0 missing)
```

**How the artifact was spotted.** Inspecting the actual "lost" strings rather
than trusting the count. Every one of them spanned a `\n\n` boundary, which is
not what real content loss looks like -- genuine loss would be arbitrary, not
perfectly correlated with paragraph breaks.

**Consequence.** When a test reports a scary number, inspect a few failing cases
before acting on it. A test that encodes a wrong model of the system produces
confident false positives, and "fixing" the code to satisfy it would have made
the chunker worse.
