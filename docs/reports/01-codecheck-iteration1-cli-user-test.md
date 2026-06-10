# CLI User Test — SelfCheck-Exec (Iteration 1)

Date: 2026-06-10
Target: `python run_codecheck.py` (`run`, `evaluate`) on MBPP+
Method: skill `cli-user-test` — drive the assembled CLI as a real user, real + adversarial
inputs. Live `run` needs `OPENROUTER_API_KEY`; the pipeline was also exercised end to end
offline by injecting a fake generator at the `CodeGenerator` seam.

## Summary

Core pipeline is correct. Two high-severity issues found that the unit suite did not catch:
a crash on complex-number inputs, and (separately) a live-API credential blocker. One
medium performance concern and one low UX wart.

## ✅ Works correctly

- **CLI surface** — `run`/`evaluate` help; all args incl. `--seed`/`--no-random`.
- **evaluate** — perfect separation → `1.0000`; single-class → `nan`; empty list → `nan`,
  n=0 (graceful, exit 0).
- **Sampling** — same `--seed` reproducible; different seeds differ; `--no-random` = first-N
  in dataset order; unseeded varies per run.
- **Core pipeline on real MBPP+ code** (fake model, no API): correct main → `is_correct=True,
  exec_score=0.000`; buggy main → `is_correct=False, exec_score=1.000`; **AUC-PR 1.0**;
  save/load roundtrip ok. Real canonical solutions executed, normalized, labeled, scored.
- **Sandbox** — `ok`/`err`/`timeout(killed)`/missing-entry handled; float-tolerance, set &
  dict order-normalization equal; both-errors → agreement (score 0); `err ≠ timeout`;
  runtime `complex` value normalizes fine.

## 🔴 Issues

**1. [High] Crash on complex-number inputs.**
`Mbpp/124` and `Mbpp/252` (2 of 378) have complex inputs (e.g. `(0.0, 1j)`). The dataset
cache write (`json.dumps(inputs)` in `load_mbpp_plus`) raises
`TypeError: Object of type complex is not JSON serializable`. Full-dataset load crashes; a
**random** `run` that samples either problem crashes at load time — nondeterministic, before
any generation. Scoring itself handles complex fine; only the JSON cache write fails.
Repro: `python -c "from codecheck.dataset import load_mbpp_plus; load_mbpp_plus()"`.

**2. [High] Live `run` blocked — 401 "Missing Authentication header."**
Config blocker, not a code bug. The `.env` key was 64-hex (`4d1c56…`), not OpenRouter's
`sk-or-v1-…` format; OpenRouter rejected it. The code built the client and surfaced the SDK
error correctly. Note: `qwen/qwen3.5-9b` may also not be a valid OpenRouter model slug.
Repro: `python run_codecheck.py run --limit 3 --n 3`.
Sub-finding: in `_cmd_run` the live `AuthenticationError` surfaces as a raw traceback (no
clean message like the missing-key path).

**3. [Med] Performance — spawn-per-input.**
~104–114 inputs *per problem*; each implementation runs each input in a fresh spawned
process. An 8-problem offline run ≈ **4,300 process spawns** (minutes). A real
`--limit 20 --n 5` ≈ 20 × 7 impls × ~110 ≈ **~15k spawns**. The execution harness, not the
API, dominates runtime at scale.

**4. [Low] `evaluate` on a missing `--results` file** dumps a raw `FileNotFoundError`
traceback instead of a clean message + exit.
Repro: `python run_codecheck.py evaluate --results /tmp/nope.json`.

## 🛠 Suggestions / fixes

- **#1** (real bug): make the cache write tolerant (`json.dumps(..., default=str)`) or skip
  caching `inputs`. Low risk — the cache is a write-only debug artifact, never read back.
- **#4**: catch missing file → friendly message + `sys.exit(1)`. Same for the live
  `AuthenticationError` in `_cmd_run`.
- **#3**: batch inputs per process (one subprocess runs all inputs for an implementation)
  instead of one-process-per-input — large speedup; defer to the scale iteration.
- **#2**: supply a valid OpenRouter key + valid model slug in `.env` (user-side).

## Status

Unit suite: 24 passing. Live AUC-PR on MBPP+: not yet observed (blocked by #2). The
"models too smart on MBPP+" question (AUC-PR → `nan` if all correct) remains open until a
real run lands.
