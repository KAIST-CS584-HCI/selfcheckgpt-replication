# CLI User Test — Iteration 2 (SelfCheck-Prompt)

Date: 2026-06-10
Target: `python run_codecheck.py run` / `evaluate`, iteration-2 surfaces (`--method`,
per-method readout, judge, mixed-method handling, `n_inputs`).
Method: skill `cli-user-test` — drive the assembled CLI as a real user, real + adversarial
inputs. Run guide: `codecheck/README.md`. Live judge path exercised with the `.env` key;
the exec/prompt pipeline also driven offline via an injected fake judge.

## Summary

Iteration-2 functionality works: `--method exec/prompt/both`, the per-method
AUC-PR + baseline + histogram readout, mixed-method tolerance, back-compat, judge
parse-failure reporting, and the live LLM-judge path all behaved as designed. Testing
surfaced **one high-severity correctness bug in the execution layer** (set/dict output
ordering is nondeterministic across spawned subprocesses, silently mislabeling correct
solutions) and one lower-severity robustness gap. Both live in the iteration-1 execution
module, not in iteration-2 code, but they directly corrupt iteration-2's scores and labels.

## ✅ Works correctly

- **CLI surface** — `run`/`evaluate` `--help`; `--method {exec,prompt,both}` exposed with
  choices; defaults to `exec`; rejects unknown (`--method ast` → argparse error).
- **Per-method evaluate readout** — for a both-method file, prints each method's trapezoidal
  AUC-PR with a per-method `(n=…)`, the shared prevalence baseline, and a 10-bin per-class
  histogram. Verified `exec`=1.0 vs `prompt`=0.29 on a crafted inverted file.
- **Mixed-method file no longer crashes** (the bug the final review caught and fixed): a file
  where only some records have `prompt` evaluates cleanly — each method is scored over only
  the records that contain it.
- **Back-compat** — legacy iteration-1 JSON (bare `exec_score` key, no `scores`/`n_inputs`)
  loads and evaluates.
- **Degenerate inputs** — single-class → `AUC-PR nan` (baseline 1.0000); empty results →
  `n=0`, `baseline nan`, exit 0; missing results file → clean `error: results file not
  found`, exit 1.
- **`n_inputs` recorded** — saved results carry the shared input-set size (e.g. `Mbpp/2` →
  111).
- **Multi-method pipeline (offline, injected fake judge)** — one run filled both `exec` and
  `prompt` scores on a real MBPP+ problem, save/load roundtrip intact.
- **Live LLM-judge path** — `run --method prompt --limit 1 --no-random --n 1` made a real
  OpenRouter judge call: `prompt=0.0` (judge ruled the sample consistent with main),
  `Judge parse failures: 0`, prompt-only run **skipped sample execution** (no subprocess
  spawns), saved + evaluated cleanly, exit 0.
- **Config blocker** — empty `OPENROUTER_API_KEY` → `error: missing OPENROUTER_API_KEY
  (see .env.example)`, clean exit.

## 🔴 Issues

**1. [High] Set/dict output ordering is nondeterministic across subprocesses → correct code
mislabeled incorrect, exec score inflated.**
Each implementation (main, samples, canonical) runs in its **own spawned subprocess**, each
with a different `PYTHONHASHSEED`. For any problem whose output ordering derives from a
`set`/`dict` (e.g. `Mbpp/2`: `return tuple(set(test_tup1) & set(test_tup2))`), the same code
produces differently-ordered tuples in different processes. Result: comparing canonical vs
canonical (identical code) yields disagreement.
Repro (identical canonical code as both main and the single sample):
- Default (randomized hashseed): `exec_score=0.027, is_correct=False` — wrong.
- `PYTHONHASHSEED=0`: `exec_score=0.0, is_correct=True` — correct.
Impact: silently corrupts **both** the consistency score (false-positive divergence on
correct code — connects to iteration-1's "false positives" finding) **and the ground-truth
label** (a known-correct solution marked incorrect) for the subset of MBPP+ problems that
return set/dict-derived collections. This distorts AUC-PR at scale with no error surfaced.
Origin: iteration-1 execution module (`codecheck/execution.py` + `normalize_output`), not
iteration-2 code, but it taints iteration-2's comparison data.

**2. [Low-Med] Execution harness silently degrades when a worker can't spawn.**
`run_batch_in_subprocess` (`codecheck/execution.py:84`, `p.start()`) uses the multiprocessing
**spawn** start method. If a worker fails to start — e.g. the harness is driven from a module
without an `if __name__ == "__main__":` guard, a frozen executable, or certain
REPL/notebook contexts — Python raises `RuntimeError: An attempt has been made to start a
new process before … bootstrapping`. The harness swallows these and returns **error-outcomes**
indistinguishable from genuine code errors, so every implementation "errors" identically →
they all "agree" and "match" → `exec_score=0.0`, `is_correct=True` for everything, with no
signal to the user. The supported CLI (`run_codecheck.py`) **is** `__main__`-guarded, so the
shipped path is unaffected; the risk is library/embedded/notebook use.
Repro: import and call `run_dataset(... run_batch_in_subprocess ...)` from an unguarded
script → flood of spawn tracebacks, yet a saved result with `exec_score=0.0, is_correct=True`.

## 🟡 Minor / observations

- **Mixed-method baseline vs per-method n.** For a heterogeneous file, the header baseline is
  computed over *all* records while each method's AUC-PR is over its own subset (shown as
  `(n=…)`). The two don't correspond for such files. Cosmetic; clean `both`/`exec` files are
  fine. Consider a per-method baseline line.
- **Empty results** prints `baseline: nan` with no methods and exit 0 — acceptable, but a
  "no results" note would read better.

## 🛠 Suggestions / fixes (none applied)

- **#1 (do before any scale run):** pin `PYTHONHASHSEED=0` in the spawned worker's
  environment so set/dict iteration order is stable across all implementations and the
  canonical labeler. (Cleaner than sort-normalizing outputs, which would wrongly equate
  genuinely order-sensitive results.) Fold into roadmap iteration-4 Exec hardening, but it
  also affects the ground-truth label, so treat it as a correctness fix, not just denoising.
- **#2:** distinguish infrastructure (spawn) failures from code errors in
  `run_batch_in_subprocess` and raise instead of folding them into "error" outcomes.
- **Minor:** add a per-method baseline to `format_evaluation`; print a "no results" line for
  empty files.

## Resolution (2026-06-10, commit `cdc124f`)

Both 🔴 issues fixed:
- **#1** — `codecheck/execution.py` now pins `PYTHONHASHSEED` (module-level `setdefault`) so
  every spawned worker hashes identically; set/dict-derived ordering is stable across
  processes. Verified: `Mbpp/2` canonical-vs-canonical → `exec_score=0.0, is_correct=True`
  with no command-line seed pin. Regression test
  `test_batch_set_ordering_is_deterministic_across_spawns`.
- **#2** — `run_batch_in_subprocess` now raises `RuntimeError` when a worker produces nothing
  and exits non-zero (spawn/bootstrap failure), instead of silently returning all-timeout.
  Regression test `test_batch_raises_on_worker_crash_before_output`.

Suite: 58 unit tests pass. The minor observations (per-method baseline, empty-results note)
were left as-is.

## Status

Iteration-2 features validated end to end, including the live judge path (1 real run).
Unit suite remains green (56). The two execution-layer issues predate iteration 2 but must be
addressed before a full-scale MBPP+ comparison, since #1 silently corrupts the very labels
and scores the AUC-PR is computed from. The plan's larger manual run
(`run --method both --limit 30 --n 5`) is still worth doing once #1 is fixed (and is the
context in which #1 would otherwise quietly skew results).
