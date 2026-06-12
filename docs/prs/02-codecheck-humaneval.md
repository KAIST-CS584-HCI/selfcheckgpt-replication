# Feat: Add HumanEval+ Dataset And A HumanEval-Tuned Judge Prompt

## Context

Checker ran on MBPP+ only. Add HumanEval+ as second dataset. Fix Prompt variant: it saturated on HumanEval+ (capable model writes near-identical samples → judge always says "consistent").

## Changes

- **HumanEval+** dataset: `run --dataset {mbpp,humaneval}` (default mbpp). Reuses whole pipeline + all four scorers.
- **HumanEval judge prompt**: makes the judge hunt for an edge-case input where two impls diverge, instead of affirming similarity. ~3x the correct-vs-incorrect gap on a real run. Oracle-free. MBPP+ keeps its old prompt.
- **`prompt` subcommand**: re-score the judge on a saved results file (like `codebert`), no regeneration.
- Bare `run` (no `--limit`) now = **entire dataset** (was silently 20).
- README: both datasets documented (HumanEval+ ships a body-only reference → loader stitches `prompt + body`). Plan docs added.

## How to run

```bash
python run_codecheck.py run --dataset humaneval --method all --output results/humaneval.json
python run_codecheck.py evaluate --results results/humaneval.json
python run_codecheck.py prompt --results results/humaneval.json   # re-score judge, no regen
```

## Notes

- `prompt` rewrites in place — use a finished/copied file, not a live run's output.
- Confident-consistent blind spot (all samples agree on same wrong answer) is inherent to consistency checking; no oracle = unfixable.
