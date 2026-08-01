# Two concurrency traps: unguarded file appends, and mock.patch

Both surfaced when preparing the API for concurrent users. Both are invisible
until something runs in parallel, and neither is visible in the file where the
bug lives.

## 1. Unguarded appends lose rows silently

`_record()` in `src/llm.py` did `open(path, "a")` per call. Measured with 8
threads writing 200 lines each:

```
expected 1600 lines, got 1434 -- 166 lost (10%)
corrupt lines: 0
```

**Zero corrupt lines** is what makes this dangerous. Each writer holds its own
file position and they overwrite one another cleanly, so the output looks
perfectly valid and simply contains less than it should. Cost accounting
under-reports and nothing complains.

The trigger was not in `llm.py` at all — FastAPI dispatches sync `def` handlers
into a threadpool, so the concurrency was injected by the framework three layers
away. "Is this thread-safe?" is not answerable from the file you are reading.

Fix: a module-level `threading.Lock` around the append. Correct within one
process; multiple uvicorn workers would still race, at which point the database
has to become the source of truth.

## 2. `unittest.mock.patch` is not thread-safe

The regression test for the above initially patched inside each worker thread:

```python
def worker():
    with patch("llm.get_client", return_value=fake_client(ok_body())):
        for _ in range(25):
            run(llm.call(...))
```

It failed with a **real HTTP 401 from OpenRouter**. `patch` swaps a module
attribute and restores it on exit; with eight threads entering and exiting
concurrently, one thread's restore put the *real* function back while another was
still running, and a live API call escaped the test.

Fix: patch once, outside the threads.

```python
with patch("llm.get_client", return_value=fake_client(ok_body())):
    # start and join all threads in here
```

**Consequence.** Any test that combines `mock.patch` with threads must apply the
patch outside them. Otherwise the test is itself racy, and the failure mode is
hitting the network — which for an LLM client means real money and a false
failure that looks like a bug in the code under test.

## Result

After both fixes: 6 concurrent `/ask` requests, 50.67s of work in 9.48s wall
clock, all HTTP 200, and all 6 ledger rows written with no loss.
