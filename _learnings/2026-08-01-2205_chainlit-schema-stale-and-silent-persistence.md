# Chainlit's published DDL is stale, and persistence failures are silent

**Symptom.** Set up Chainlit's `SQLAlchemyDataLayer` against Neon using the
`CREATE TABLE` statements from
<https://docs.chainlit.io/data-layers/sqlalchemy>. The UI worked perfectly:
messages rendered, `cl.Step` trees expanded, thumbs appeared. But only
`user_message` / `assistant_message` rows landed in `steps` — the `Retrieval` and
`Generation` steps carrying all our token and cost metadata were **missing**, with
no error anywhere in the UI.

**Cause, two compounding.**

1. **The documented schema is behind the shipped package.** chainlit 2.11.1
   writes an `autoCollapse` column that the docs' DDL never creates:

   ```
   asyncpg.exceptions.UndefinedColumnError:
     column "autoCollapse" of relation "steps" does not exist
   ```

   Diffing `chainlit.step.StepDict` against the created table showed three
   fields missing: `autoCollapse`, `icon`, `feedback`. Only `autoCollapse` is
   written on every insert, so it broke everything.

2. **The failure is silent by design.** `Step.send()` persists via
   `asyncio.create_task(data_layer.create_step(...))` — fire-and-forget — inside
   a `try` that only calls `logger.error` unless `fail_on_persist_error` is set.
   So every insert failed, the UI stayed flawless, and the only trace was a
   `WARNING` line in the server log.

**Fix.** Derive the schema from the installed package, not the docs:

```python
from chainlit.step import StepDict
sorted(StepDict.__annotations__.keys())
```

then `ALTER TABLE steps ADD COLUMN IF NOT EXISTS "autoCollapse" BOOLEAN` (and
`"icon" TEXT`). The quoted camelCase names are load-bearing — unquoted
identifiers get folded to lowercase by Postgres and stop matching Chainlit's
queries.

**Consequences.**

- After wiring any Chainlit data layer, **verify a row actually lands** before
  believing it works. `SELECT count(*) FROM steps` is the check; the UI is not.
- Consider `fail_on_persist_error=True` in development so this crashes loudly
  instead of degrading into a chat app that quietly remembers nothing.
- Generalise: when a library's docs carry a schema, the schema is documentation
  and drifts like documentation. The installed code is the source of truth.

**Unrelated gotcha found the same session:** `pkill -f "chainlit run"` does not
match a process started as `python -m chainlit run` on Windows. The old process
kept port 8001 and kept serving stale code while a "restarted" server appeared
healthy — and because the old process still held the redirected log file open, it
kept writing to `logs/chainlit.log` after truncation, so the log looked like the
new process was handling requests. Kill by PID from
`netstat -ano | grep :8001` and confirm the port is free before restarting.
