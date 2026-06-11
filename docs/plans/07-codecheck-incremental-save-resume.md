---
type: plan
status: approved
created: 2026-06-11
source: brainstorming session (incremental save + resume)
issue: docs/issues/01-run-aborts-on-exhausted-api-call.md
---

# Incremental save + resume

## Context

Results are written only at the end of a run, so an interrupt (Ctrl-C, OOM, kill) before
the final save loses every completed problem — exactly the pain behind the 1h-lost run in
`docs/issues/01`. We want each problem persisted the moment it finishes, and a re-run of
the same command to continue where it left off rather than redo finished problems.

Decisions (from brainstorming): **JSONL** append-only format; **auto-resume** when the
output file already exists; identity key is `task_id`.

## Design

**Format — JSONL.** One `CodeResult` JSON object per line. Appending is O(1) and a torn
final line (crash mid-write) is simply skipped on load.

**Pipeline (`codecheck/pipeline.py`).**
- `append_result(result, path)` — ensure parent dir, open in append mode, write
  `json.dumps(result.to_dict(), ensure_ascii=False) + "\n"`, close (flush per problem).
- `save_results(results, path)` — rewrite the whole list as JSONL via the existing atomic
  temp-replace. Kept for batch writes and tests.
- `load_results(path)` — **tolerant**: if the first non-space char is `[`, parse as a
  legacy JSON array (keeps existing committed `.json` artifacts and offline recompute
  working); otherwise parse line by line, skipping blank/unparseable lines (skips a torn
  trailing line). Returns `list[CodeResult]`.
- `run_dataset(..., on_result=None)` — after each successful result (and after the readout
  line), call `on_result(result)` if provided. Keeps the loop decoupled from file I/O; the
  skip-on-failure and end-of-run summary behavior is unchanged.

**CLI (`run_codecheck.py` `_cmd_run`).**
- After resolving problems, if `--output` exists: `existing = load_results(output)`;
  normalize the file once with `save_results(existing, output)` (legacy array → clean
  JSONL, drops a torn line); `done = {r.task_id for r in existing}`; filter problems to
  those whose `task_id` not in `done`. Print `resuming: <len(done)> done, <remaining> remaining`.
- Call `run_dataset(remaining, ..., on_result=lambda r: append_result(r, output))` so each
  finished problem is persisted immediately.
- If nothing remains, skip the run. Drop the end-of-run `save_results` call (the file is
  already complete via append). Final message: total results in file (`len(existing)+new`).
- `DEFAULT_OUTPUT` → `output/codecheck.jsonl`. `--output` extension is not enforced.

**Unchanged.** `evaluate` (reads via the now-tolerant `load_results`); the `CodeResult`
schema; the failure-skip + summary logic in `run_dataset`.

## Critical files

- `codecheck/pipeline.py` — `append_result` (new), `save_results`/`load_results` → JSONL
  (tolerant load), `run_dataset` gains `on_result`.
- `run_codecheck.py` — resume/normalize/filter in `_cmd_run`, `on_result` wiring, drop
  final save, `DEFAULT_OUTPUT` → `.jsonl`.
- Tests: `tests/test_codecheck_pipeline.py` (append, JSONL roundtrip, tolerant load incl.
  legacy array + torn final line, `on_result` called per success). CLI resume-filter
  covered by a small pipeline-level/`_cmd_run` check.

## Reuse

- Existing atomic temp-replace pattern in `save_results` (`codecheck/pipeline.py`).
- Existing `CodeResult.to_dict` / `from_dict` (`codecheck/models.py`).
- Existing per-problem loop + failure handling in `run_dataset`.

## Verification

- Unit: `pytest tests/test_codecheck_pipeline.py -q`
  - `append_result` then `load_results` returns the appended results in order.
  - `save_results` (JSONL) → `load_results` roundtrip equals the input.
  - `load_results` reads a hand-written legacy `[...]` array file.
  - `load_results` skips a torn/blank final line without raising.
  - `run_dataset(..., on_result=sink)` calls `sink` once per successful problem, not for
    skipped failures.
- Full suite: `pytest tests/ -q -k codecheck` (green; `evaluate` still reads results).
- Manual (needs `OPENROUTER_API_KEY`):
  1. `python run_codecheck.py run --method ast --limit 5 --output /tmp/r.jsonl` — confirm
     `/tmp/r.jsonl` gains a line per problem during the run.
  2. Interrupt a longer run with Ctrl-C, then re-run the same command — it prints
     `resuming: N done, M remaining` and only runs the remainder.
  3. `python run_codecheck.py evaluate --results /tmp/r.jsonl` — reads the JSONL fine.
