# Deferred

Parked deliberately, so it does not get lost. Each item says what is wrong and
what would make it worth doing.

---

## Eval set validity

### ~~Questions with unresolved deictics~~ — DONE 2026-08-02

Eight questions rewritten by hand to name their subject; the generator prompt now
requires self-containment. Final set: median overlap 0.46, max 0.65, none above
`curate.py`'s rejection threshold. The five questions still containing a
demonstrative all have their antecedent inside the question itself
(`նրանք` = ԲՀԿ, `այդ երկրի` = Իտալիա, `նա` = Խուդավերդյան) and are fine.
See DECISIONS.md #20.

### Harder questions — generator works, 9 candidates awaiting a read

`eval/generate_hard_candidates.py --pairs 12 --ban-terms` drafts questions
against a distractor article with the source's distinctive vocabulary forbidden.
That combination is what works; the distractor alone did nothing. Full numbers in
[`_learnings/2026-08-02-1610_lexical-overlap-is-the-dial-on-eval-difficulty.md`](_learnings/2026-08-02-1610_lexical-overlap-is-the-dial-on-eval-difficulty.md).

**Nothing has been merged into `questions.toml`.** 9 screened candidates sit in
`eval/hard_candidates_screened.jsonl` and need an Armenian read before adoption —
auto-accepting generated questions measures the generator, not the system.
Known nits in them: #5 ends with a Latin `?` instead of `։`, #9 is a leading
either/or.

Two things to fix before the next batch:
1. **Generate `must_contain` in a separate pass** that can see the banned words.
   Banning the article's vocabulary degraded the assertions (5 of 15 rejections),
   because those strings need exactly the words the question may not use.
2. **Enforce self-containment mechanically**, the way overlap is enforced — 7 of
   24 were rejected for a bare deictic despite the prompt forbidding it. Same
   lesson as the paraphrase instruction: check it, do not ask for it.

Also worth knowing: the screen removed hardness along with defects — retriever
disagreement was 6/24 in the raw batch and 1/9 among survivors.

### `answer_must_contain` is morphology-blind

Literal substring matching against an agglutinative language. Measured: of 10
failures in the last run, most were inflection (`Կառավարությունից` vs the
asserted `Կառավարությունը`) or synonym (`արդար չէ` vs `արդարացի չէ`), not wrong
answers. Real pass rate is above the reported 63%.

Options, cheapest first:
1. **Rewrite the assertions** to stems, numbers and proper nouns that do not
   decline. No new dependency. Roughly 20 strings to revise by hand.
2. **Normalise before comparing** — strip a list of Armenian case suffixes.
   Fragile; suffix rules were already rejected for BM25 (DECISIONS.md #16).
3. **Drop the check** and rely on the LLM judge. Loses the only LLM-free signal.

Leaning (1). Not urgent while the report says out loud that failures need
reading before they are believed.

---

### No multi-turn cases in the eval set

Query rewriting now ships (DECISIONS #21) and **nothing in `eval/questions.toml`
exercises it** — every case is single-turn, and `run_eval.py` sends no history.
That was fine when the system was stateless; it is now a real blind spot.

Verified by hand at the time of the change: the "of those" case resolves
correctly (118 → 310) and two already-standalone questions pass through
unchanged. That is three data points, not a measurement.

**What a multi-turn bucket needs:** a `history` field per question, an assertion
on the *rewritten* query rather than only the answer, and cases for the failure
modes that actually matter —
1. a genuine follow-up that must inherit an entity from turn 1,
2. an already-standalone question asked mid-conversation (must be left alone),
3. a topic switch (must NOT inherit the previous subject),
4. a follow-up whose referent is genuinely ambiguous (must not be invented).

## Measurement not yet taken

- **Grounding across the 27 factual questions.** The pinned judge
  (`openai/gpt-5.2-pro`, ~$3.28 for a full run, cached by
  question+answer+excerpts+model) has never been run. Everything in
  `eval/RESULTS.md` under "Grounding" is a hand-graded sample.
- **A recorded run at `max_distance = 0.55`.** The value was picked from an
  offline sweep; both runs in `eval/results.jsonl` are at 0.50.
- **Hybrid retrieval (BM25 + dense).** The intended destination per
  DECISIONS.md #18; the eval set exists now and should decide whether it beats
  dense alone.

## Operational

### No HTTPS — deployed over plain HTTP on the bare IP

**Decided 2026-08-02, knowingly, because this is a short-lived demo.** Let's
Encrypt does not issue certificates for IP addresses, and there is no domain
yet, so Caddy runs on `:80` with no TLS.

**What is exposed:** the `APP_PASSWORD` and every question and answer travel in
cleartext. Anyone on the path — shared wifi, hotspot, ISP — can read them or
modify the page in flight. The browser shows "Not secure".

**Mitigation now:** use a throwaway `APP_PASSWORD` that is not reused anywhere.

**The fix, when the demo becomes anything more than a demo:**
1. Register a domain (~$1-3/yr for `.xyz`), or use a free `sslip.io` name —
   `157-90-1-2.sslip.io` resolves to `157.90.1.2` with no registration at all.
2. Point an `A` record at the server IPv4.
3. Set `DOMAIN=` in `.env`, then `docker compose up -d`.

No rebuild and no code change: `docker-compose.yml` derives
`SITE_ADDRESS: ${DOMAIN:-:80}`, so setting `DOMAIN` switches Caddy to automatic
Let's Encrypt HTTPS on its own. Verified both branches resolve correctly with
`docker compose config`.

**Trigger to do this:** anyone other than the author uses the URL, or it stays
up longer than the demo it was stood up for.

- **No rate limiting and no per-user cost cap.** `APP_PASSWORD` gates the door
  but does not identify anyone. Must exist before the URL is shared.
- **Only 94 of 5,836 indepth articles indexed.** The Colab path makes a full
  index minutes rather than ~28h on this CPU.
- **Cost ledger lock is per-process.** Multiple uvicorn workers would race
  again; at that point cost accounting belongs in Postgres.
