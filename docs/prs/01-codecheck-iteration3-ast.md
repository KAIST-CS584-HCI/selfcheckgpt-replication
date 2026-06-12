# Feat: Add SelfCheck-AST Code-Hallucination Variant (Iteration 3)

## Context

SelfCheck for code had two consistency signals (Exec, Prompt). This adds the third,
**SelfCheck-AST**: it scores how structurally consistent a model's main implementation is
with its own samples, no API calls needed. Completes the Exec / Prompt / AST set so all
three can be compared on identical data.

## Changes

- Added the **SelfCheck-AST** scorer: parses each implementation to an AST and measures
  structural divergence between the main answer and its samples. Rename- and
  literal-invariant, scored on the same `[0,1]` scale as Exec/Prompt (higher = more likely
  incorrect).
- Two selectable AST metrics via `--ast-metric`: `jaccard` (default, bag-of-node-types) and
  `ted` (tree edit distance, via the `zss` dependency).
- Wired AST into the run/evaluate pipeline and CLI: `--method ast` and `--method all`
  (Exec + Prompt + AST three-way); dropped the now-redundant `--method both`.
- Added `-v/--verbose` and richer API logging: every call logs latency, finish reason, and
  token counts at DEBUG; truncated or empty responses warn by default.
- Hardened robustness: blank/non-Python implementations are treated as parse failures (not
  silently scored), and stuck execution/API workers are bounded so a run can't hang.

## Results

- On the validation sample, all three methods beat the prevalence baseline (Exec strongest,
  AST a complementary third signal).
- Jaccard-vs-TED comparison found TED does **not** improve discrimination on MBPP+, so
  `jaccard` is the default. TED stays available. See `docs/reports/09`.

## How to run

```bash
# three-way comparison on one sample
python run_codecheck.py run --method all --limit 30 --n 5 --seed 1 --timeout 5 \
  --output output/all.json
python run_codecheck.py evaluate --results output/all.json

# AST only, with tree edit distance instead of the default
python run_codecheck.py run --method ast --ast-metric ted --limit 30 --n 5
```

## Notes

- `zss` is a new dependency (pure-Python, AST tree edit distance).
- The 3 failing `test_score_cli.py` tests are pre-existing (missing `groq` module in the
  unrelated WikiBio path), not from this branch.
