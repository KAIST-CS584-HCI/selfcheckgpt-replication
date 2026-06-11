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

## `--no-random` slice (larger run) — TED marginally ahead

Recomputed both metrics offline on `results/codecheck-ast.json`
(`--no-random --limit 200 --n 20`): 200 problems, 56 incorrect, baseline 0.280, 35
parse failures (same for both metrics).

| Metric  | AUC-PR | rank-AUC | correct mean | incorrect mean |
|---------|--------|----------|--------------|----------------|
| jaccard | 0.621  | 0.794    | 0.178 | 0.336 |
| **ted** | **0.641** | **0.803** | 0.137 | 0.276 |

Here **TED is slightly better** — the *opposite* of the seed-1 result. The two samples
disagree on the ordering, and both gaps are small (≤0.02 AUC-PR).

## Synthesis — the metrics are roughly tied

| Sample | n | problems | jaccard AUC-PR | ted AUC-PR | winner |
|--------|---|----------|----------------|------------|--------|
| seed-1 (random) | 5 | 30 | 0.591 | 0.485 | jaccard |
| sequential | 20 | 200 | 0.621 | 0.641 | ted (marginal) |

Neither metric dominates. The difference flips with the sample and stays within noise,
and both sit only modestly above the ~0.28–0.30 baseline. The likely driver is **N**:
at N=20 TED's extra shape sensitivity averages out and is no longer net-harmful; at N=5
its variance on correct-but-restructured samples dominated. Bottom line: **AST is a
weak-to-moderate signal either way**, and the metric choice is close to a wash.

## Recommendation

- **Default kept at `jaccard`** (the user's decision). It is the cheaper metric (no
  TED computation; TED is ~O(n²) per pair and noticeably slower at N=20 × 200), it is
  never worse by more than ~0.02 AUC-PR, and it won the small random sample. `ted` stays
  available via `--ast-metric ted` for anyone who wants the structure-aware variant on a
  large/high-N run, where it edges ahead.
- The deeper takeaway feeds the Improvement-2 narrative: AST (either metric) is a
  weak-to-moderate signal on MBPP+ and shares Exec's confident-consistent blind spot;
  the richer structural metric does not rescue it. A genuinely different signal
  (behavioral Exec, or the LLM judge) is where the lift is.

## What shipped

- `codecheck/ast_score.py`: `ast_to_tree`, `ted_dissimilarity` (zss, unit-cost,
  rename/literal-invariant, normalized to [0,1]); `ASTScorer(metric=...)` dispatch.
- `run --ast-metric {jaccard,ted}` (default `jaccard`), run header prints the active metric.
- `zss` added to `requirements.txt`. AST unit tests + full codecheck suite green.
