# One exhausted API call aborts the entire run (no per-problem isolation or incremental save)

## Summary

When a single problem's generation exhausts all API retries, `chat_with_retries`
raises `APIRetriesExhausted`, which propagates up and kills the whole `run_dataset`
loop. Results are only saved at the end, so all completed problems are lost.

## Observed

`--method ast --limit 200 --n 20` crashed at **123/200 after ~1h05m**:

```
ERROR API call failed after 4 attempts (TimeoutError)
...
codecheck.api_retry.APIRetriesExhausted: TimeoutError()
```

Each retry hit the 60s wall-clock cap (`call_timeout`); after 4 attempts the
exception bubbled through `generate` → `score_problem` → `run_dataset` → process exit.
122 scored problems were discarded.

## Impact

One unrecoverable API stall on a single problem wastes an entire long run. The longer
the run, the more it hurts.

## Direction (not fixing now)

- Isolate per-problem failures: catch `APIRetriesExhausted` in `run_dataset`, record the
  problem as failed/skipped, continue.
- And/or save results incrementally so a crash keeps prior progress (resume support).

## Minor

- Log says `retrying 4/3` (attempt counter off-by-one in the message; cosmetic).
