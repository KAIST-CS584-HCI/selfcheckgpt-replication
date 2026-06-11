---
type: plan
status: approved
created: 2026-06-11
source: brainstorming session (global error handling for run_dataset)
issue: docs/issues/01-run-aborts-on-exhausted-api-call.md
---

# Continue-on-failure for `run_dataset`

## Goal

One problem's failure (e.g. an exhausted-retry API timeout) must not abort the whole
run. Exceptions propagate up unchanged from the low/mid-level functions (generation,
`api_retry`, execution) and are caught once at the per-problem orchestration boundary,
so the run skips the failed problem and continues.

## Decisions (from brainstorming)

- **Single capture point: the per-problem loop in `run_dataset`.** Catching only at the
  CLI entry point cannot resume the loop, so the boundary is the loop body. This is the
  orchestration layer, not a low/mid function — generation/api_retry/execution stay
  free of try/except and just let exceptions propagate.
- **Catch broad `Exception`**, but re-raise `KeyboardInterrupt` / `SystemExit` so the
  user can still abort. Any single-problem failure (API exhaustion, execution glitch,
  unexpected bug) skips that problem rather than killing the run.
- **Skip + end-of-run summary.** A failed problem is logged with its full traceback,
  omitted from results, and its `task_id` collected. After the loop, a summary line
  reports `N/total failed: [task_ids]`. Results file contains only successful problems;
  `evaluate` and the loaders need no change.
- **Continue-on-failure only.** No incremental/periodic save — the broad catch already
  guarantees the run reaches its final save with every successful problem.

## Architecture

Only `codecheck/pipeline.py` changes.

```python
logger = logging.getLogger("codecheck.pipeline")  # propagates to the "codecheck" run handler

def run_dataset(...):
    problems = list(problems)
    total = len(problems)
    results, failed = [], []
    for i, problem in enumerate(tqdm(problems, desc="codecheck"), start=1):
        started = time.monotonic()
        try:
            result = score_problem(problem, generator, harness, n_samples, timeout,
                                   methods, judge, ast_scorer)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logger.exception("problem %s failed; skipping", problem.task_id)
            failed.append(problem.task_id)
            continue
        elapsed = time.monotonic() - started
        ...existing readout line...
        results.append(result)
    if failed:
        logger.warning("%d/%d problems failed and were skipped: %s",
                       len(failed), total, ", ".join(failed))
    return results
```

- `logger.exception` preserves the traceback so genuine bugs stay visible; the failure
  is not silently swallowed.
- `run_dataset` keeps its current return type (list of successful `CodeResult`). The
  CLI's existing "Saved N results" already reflects successes.

## No changes to

`generation.py`, `api_retry.py`, `execution.py`, `score_problem`, `models.py`,
`evaluate.py`, and the CLI entry point (its outer `AuthenticationError` catch stays for
fatal config errors).

## Testing

- **Continues past a failing problem:** a stub generator that raises on the 2nd of 3
  problems → `run_dataset` completes, returns 2 results, the 2 successful task_ids
  present, the failed one absent.
- **Failure is reported:** the failed task_id appears in a logged warning (caplog).
- **KeyboardInterrupt propagates:** a stub that raises `KeyboardInterrupt` → `run_dataset`
  re-raises, does not swallow.

## Risks / open

- A systematically failing run (every problem errors) now produces an empty result file
  instead of crashing. The end-of-run summary makes that visible (`N/N failed`), and
  `evaluate` on an empty file already degrades gracefully.
