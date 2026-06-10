# CLI User Test — Live Run (Iteration 1)

Date: 2026-06-10
Target: `python run_codecheck.py run` / `evaluate` on MBPP+
Method: skill `cli-user-test`. Follows report 01 (offline); this run exercises the **live
API path** that 01 was blocked on, after a valid OpenRouter key was supplied.
Run guide: `codecheck/README.md` (gate satisfied — no guessing needed).

## Inputs / Outputs / Artifacts

- **Input (to the program):** `run --limit 1 --n 1 --seed 0 --timeout 5`. With `--seed 0`
  this selected `Mbpp/435` ("last digit of a number"). Its prompt was sent to
  `qwen/qwen3.5-9b` via OpenRouter: 1 main impl at T=0, 1 sample at T=1.
- **Output (from the program):** one `CodeResult` row — `task_id, is_correct, exec_score,
  main_code, sample_codes` — written as JSON; then `evaluate` printed the AUC-PR.
- **Artifact:** the raw result is saved next to this report as
  `02-codecheck-iteration1-live-result.json` (re-runnable through `evaluate`).

Format note: raw runs are kept as **JSON** (program-native, machine-readable, re-evaluatable);
this human report is **MD**.

## Summary

The live path works end to end. The method caught a real bug on the generated code.
One high-severity throughput problem makes scale runs infeasible as-is.

## ✅ Works correctly

- **End-to-end live run** — generation → execution → scoring → labeling → save all
  functioned with a valid `sk-or-v1…` key. `Saved 1 results`.
- **Real generated code executed** — `Mbpp/435` → `def last_Digit(n): return n % 10`.
- **Caught a genuine bug** — flagged `is_correct=False`, `exec_score=0.358`. `n % 10`
  diverges from the canonical on negative inputs in `plus_input` — a real behavioral
  difference, exactly what SelfCheck-Exec is meant to detect.
- **Reproducible scoring** — re-scoring the same generated codes offline reproduced
  `exec_score=0.358` exactly (deterministic given the codes).
- **evaluate** — single-class result → `nan`, `n=1`, exit 0 (clean).

## 🔴 Issues

**1. [High] Throughput — 4m53s for ONE problem at n=1.** Measured attribution:
- **~274s API latency (dominant).** Generation calls are **sequential** (`generate` loops
  `_complete` for main + each sample) and `qwen/qwen3.5-9b` is slow (~135s/call).
- **~19s execution.** 318 process spawns (106 inputs × 3 implementations) — the known
  spawn-per-input cost (report 01, #3).

Projection: n=5 ≈ ~13 min/problem; `--limit 20 --n 5` ≈ **4–5 hours**; paper-scale
(full set, n=20) is **infeasible** as-is. Not a config bug — the model is valid and
responds, just slow and serialized.

## 🛠 Suggestions / fixes (none applied)

- **Parallelize generation** (biggest win): fire the N+1 calls concurrently
  (`ThreadPoolExecutor`), as `replication/score/prompt.py` already does. Cuts generation
  wall-time ~N×. *(Deferred per maintainer: not needed yet.)*
- **Batch exec inputs per process** (#3): one subprocess runs all inputs for an
  implementation → ~19s drops to sub-second.
- **Faster/cheaper sample model** or smaller default N for smoke runs.

## Status

Live path validated on 1 problem. Larger live runs intentionally not done — at ~5
min/problem they add cost/time without new signal beyond the throughput finding. The
"models too smart on MBPP+" question is still open (need a multi-problem run once
throughput is addressed), though the first real problem already produced a *wrong*
implementation, which is encouraging for signal.
