# SelfCheck-AST — Jaccard vs Tree Edit Distance

Date: 2026-06-10
Plan: `docs/plans/05-codecheck-ast-ted-metric.md`
Method: TDD, 5 commits. Both metrics computed **offline from identical generated code**
(AST scoring is pure-local), so the comparison is apples-to-apples on the same
implementations.

## Question

Does the structure-aware TED metric (Zhang-Shasha tree edit distance) separate correct
from incorrect mains better than the count-based bag-of-node-types Jaccard? TED was made
the default on the hypothesis that it would.

## Result — seed-1 sample (30 problems, n=5, the iter-3 data)

Recomputed both metrics on `docs/reports/07-codecheck-iteration3-ast-result.json`.
Baseline (incorrect prevalence) = 0.300. 0 parse failures either metric.

| Metric  | AUC-PR | rank-AUC | correct mean | incorrect mean | separation |
|---------|--------|----------|--------------|----------------|------------|
| jaccard | **0.591** | **0.778** | 0.178 | 0.355 | **+0.177** |
| ted     | 0.485  | 0.751    | 0.149 | 0.276 | +0.127 |

**TED is worse, not better**, on every measure: lower AUC-PR, lower rank-AUC, and a
*smaller* gap between the classes. TED also compresses both class means downward.

## Why structure-awareness hurts here

TED is more sensitive to exact tree shape than Jaccard. On MBPP+ that sensitivity works
*against* the signal:

- **Correct mains take more false-positive pressure.** Correct code admits many valid
  shapes; a correct sample that restructures the same logic (different nesting, helper,
  comprehension vs loop) is structurally far under TED but near under Jaccard's node
  counts. TED inflates the score on correct mains → less separation.
- **Incorrect mains don't diverge enough structurally.** This is the iteration's
  standing hypothesis confirmed from another angle: a consistently-wrong main has
  samples that are *structurally similar to it* (same wrong shape), so neither metric
  flags them — and TED's extra shape sensitivity adds noise on the correct side without
  adding signal on the incorrect side.

Net: on MBPP+, structural consistency tracks correctness only weakly, and a *coarser*
structural metric (Jaccard) is the better discriminator because it ignores the
correct-but-restructured variation that TED penalizes.

## `--no-random` slice (your intended protocol)

<!-- FILLED WHEN results/codecheck-ast.json (live --no-random --limit 200 --n 20) LANDS;
     recomputed offline the same way. -->
Pending the in-flight `--no-random --limit 200 --n 20` run. Will be recomputed offline
(both metrics, same codes) and appended. The seed-1 result above is the primary,
apples-to-apples evidence; the easy low-numbered slice is expected to compress both
metrics further (fewer positives, more consistent samples), not reverse their ordering.

## Recommendation

- **Reconsider the TED default.** The evidence says `jaccard` discriminates better on
  this data. Either flip the default back to `jaccard`, or keep `ted` available but
  default to `jaccard`. (Decision deferred to the user — TED-as-default was the stated
  directive; this report is the evidence to revisit it.)
- **Both metrics stay selectable** via `--ast-metric`, so this comparison is cheap to
  re-run on any future sample (offline, zero API cost).
- The deeper takeaway feeds the Improvement-2 narrative: AST (either metric) is a
  weak-to-moderate signal on MBPP+ and shares Exec's confident-consistent blind spot;
  the richer structural metric does not rescue it. A genuinely different signal
  (behavioral Exec, or the LLM judge) is where the lift is.

## What shipped

- `codecheck/ast_score.py`: `ast_to_tree`, `ted_dissimilarity` (zss, unit-cost,
  rename/literal-invariant, normalized to [0,1]); `ASTScorer(metric=...)` dispatch.
- `run --ast-metric {jaccard,ted}` (default `ted`), run header prints the active metric.
- `zss` added to `requirements.txt`. 24 AST unit tests; full codecheck suite 107 green.
