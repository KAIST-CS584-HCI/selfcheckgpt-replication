# Feat: Add CodeHaluEval Dataset And Harden Offline Scoring Commands

## Context

Third code-hallucination benchmark. MBPP+/HumanEval+ = function-call; CodeHaluEval = whole-program stdin→stdout (Codeforces-style), built to induce hallucinations → real incorrect class to stress detectors. Also fixes offline re-score commands that ran silent, looked frozen.

## Changes

- **`--dataset codehalu`**: runs CodeHaluEval stdin→stdout problems end to end (gen, exec, 4 scorers, evaluate).
- **Whole-program harness**: pipe each test case stdin into fresh Python subprocess, capture+normalize stdout, compare behavior across impls.
- **Program-oriented judge**: stdin/stdout variant of prompt judge, auto-selected for this dataset.
- **Ground truth ships with data**: each case carries expected stdout → correctness labeled direct, reference solution never run.
- **Reasoning on by default** for CodeHaluEval generation: competitive problems much harder than one function, model reasons to produce usable programs.
- **Malformed tasks skipped**, not fatal: one bad record no longer aborts whole load.
- **`prompt` + `codebert` now show progress bar + save per result**: were silent, saved only at end → long pass looked stalled, Ctrl-C lost all work. Interrupt now keeps finished scores.
- **`codebert` skips already-scored results** by default; `--recompute` rescores all (cheap resume of partial pass).

## How to run

```bash
# small end-to-end check
python run_codecheck.py run --dataset codehalu --limit 2 --n 3 --method all --timeout 5 --output results/che.json
python run_codecheck.py evaluate --results results/che.json

# fill missing code_bert (resumable)
python run_codecheck.py codebert --results results/run.json
```

## Notes

- CodeHaluEval stored `solutions` often partial/absent → reference-only; correctness from shipped expected.
- `prompt` noisy on whole-program code (judge keys on impl diffs); `exec` strongest. `code_bert` stays saturated (near-baseline), as on other datasets.
