# A constraint measured on the wrong OS picked our hosting for us

Peak memory was the binding constraint for deployment. It was measured on the
Windows dev laptop at **1793 MB**, which set a hard "needs ≥ 4 GB" requirement
and eliminated every 2 GB candidate from the provider comparison.

Measured on the actual deployed container (Ubuntu 26.04, Docker, cx23):

```
after 1 query   782 MiB
after 2         785 MiB
after 3         788 MiB
after 4         792 MiB     <- stable, ~3 MiB drift per query
```

**~790 MB, not 1793 MB.** A 2.3x overestimate.

## The tell was already in our own notes

DEPLOY.md said, correctly:

> Thread count does not help (1791 MB at 1 thread vs 1792 at 4) — the memory is
> torch's **forward-pass arena**, not the weights, which load in 630 MB.

The diagnosis was right and the conclusion was never drawn: **an allocator arena
is exactly the kind of thing that differs between platforms.** Windows and glibc
malloc have different arena sizing and different return-to-OS behaviour. The one
component identified as dominant was also the one component least likely to
transfer. On Linux it is 630 MB of weights plus ~160 MB overhead, flat.

## What it cost

The 1793 MB figure ruled out Fly.io (2 GB, ~$10.70), Render (2 GB), and every
other 2 GB tier on memory grounds. All of them would have fit.

It did **not** change the final answer, by luck: cx23 at €7.13/mo is the cheapest
plan Hetzner offers at any size, and the 2 GB cpx12 is €13.67 — *more expensive*
for less RAM. So the decision survived a wrong premise. That is not vindication;
the comparison table in DECISIONS #19 was simply wrong in a way that happened not
to flip the ranking.

## The rule

**Measure a deployment constraint on the deployment platform**, or label the
number with the platform it came from and treat it as an estimate until
re-measured. Specifically suspect cross-platform transfer for:

- allocator behaviour and RSS (arenas, fragmentation, return-to-OS)
- anything involving `torch`, BLAS thread pools, or memory-mapped weights
- default wheel contents — the same `pip install torch` gives a CPU-only build on
  Windows and a CUDA build on Linux, which we already got bitten by

**A number with no platform attached is not a measurement.** Every row in a
sizing table should say where it was taken.

Related: [[2026-08-02-1214_two-sources-that-must-agree]] — same family. There, two
artefacts had to agree and nothing enforced it. Here, a number had to be valid on
a platform it was never taken on, and nothing flagged it.
