# Three-way Exec / Prompt / AST comparison (Iteration 3)

Date: 2026-06-10
Artifact: `output/iter3-all.json` (also serves as the iteration-2.5 validation gate —
see note below). Run log: `output/iter3-run.log`.

## Setup

One `--method all` run, capable model `qwen/qwen3.5-9b`, random full-set sample
(`--limit 30 --seed 1`), `--n 5`, `--timeout 5`. This single run is a **superset of
the iter-2.5 gate run** (`--method both` on the same seed-1 sample): it produces the
trustworthy post-hashseed-fix Exec/Prompt numbers the gate was meant to establish
*and* the new AST column, all on identical generated code, in one pass.

> **Caveat on cross-run probe comparison.** `--seed` fixes *problem selection* only;
> code generation is stochastic (temperature sampling, no generation seed). So the
> specific probe scores in the iter-2.5 plan (e.g. `Mbpp/237` exec=0/prompt=0) came
> from a different generation and do **not** reproduce here. This report uses *this
> run's own* confident-consistent cases, not the plan's stale ones.

## Headline numbers

| Method | AUC-PR (detect incorrect) | vs baseline 0.300 |
|--------|---------------------------|-------------------|
| exec   | **0.696**                 | +0.40             |
| ast    | **0.591**                 | +0.29             |
| prompt | **0.534**                 | +0.23             |

n=30, 9 incorrect (30% prevalence). Judge parse failures: 0. AST parse failures: 0.

Per-class score separation (mean / median):

| Method | correct | incorrect |
|--------|---------|-----------|
| exec   | 0.062 / 0.000 | 0.391 / 0.200 |
| prompt | 0.229 / 0.200 | 0.511 / 0.600 |
| ast    | 0.178 / 0.127 | 0.355 / 0.242 |

## Findings

1. **AST adds real signal, above the baseline and above Prompt on this sample.**
   AUC-PR 0.591 vs the 0.300 floor, and it edges out Prompt (0.534). Exec stays the
   strongest single signal (0.696). The bag-of-node-types Jaccard metric **does
   discriminate** — the coarseness risk in the plan did not sink the MVP.

2. **Clean low-end specificity.** All 8 implementations in the AST `[0.0,0.1)` bin are
   correct; every one of the 9 incorrect mains scores `ast ≥ 0.156`. AST never scored
   an incorrect main as near-zero — unlike Exec (2 incorrect in `[0.0,0.1)`) and
   Prompt (2 incorrect in `[0.0,0.1)`).

3. **Blind-spot check — AST gives a *small* lift where Exec+Prompt fully miss.** The
   two genuine confident-consistent misses this run (both `exec=0.000` *and*
   `prompt=0.000`):
   - `Mbpp/99` (incorrect) → `ast=0.208`
   - `Mbpp/310` (incorrect) → `ast=0.167`

   Both are non-zero, so AST **partially refutes** the plan's hypothesis that
   structure shares Exec's confident-consistent blind spot entirely. But the lift is
   weak: 0.17–0.21 ranks them low-middle, not strongly flagged. AST sees *some*
   structural wobble the samples hide behaviorally, just not much.

4. **AST tracks neither Exec nor Prompt cleanly — it is a third axis.**
   - `Mbpp/237` (incorrect): exec=0.200, prompt=0.400, **ast=0.705** — AST's top
     score, catches an impl the other two rate mid-low.
   - `Mbpp/785` (incorrect): prompt=1.000 catches hard; exec=0.374, **ast=0.406**
     tracks Exec's moderate read, not the judge.
   - `Mbpp/430`, `Mbpp/577` (incorrect): all three high — genuine divergence, AST
     sanity holds (ast=0.591, 0.550).

5. **AST's cost is false positives (its main weakness).** Four *correct* mains score
   `ast ≥ 0.3`: `Mbpp/274` (0.629), `Mbpp/19` (0.400), `Mbpp/20` (0.364), `Mbpp/3`
   (0.343). Correct code admits many structurally different-but-valid
   implementations; bag-of-node-types divergence cannot tell "different correct shape"
   from "wrong shape." Same family as Exec's adversarial-input false positives.

## Iteration-2.5 gate verdict (folded in)

- **Exec signal survives the hashseed label fix — and is stronger here:** AUC-PR
  0.696 vs the tainted iter-1 0.4545. (Different sample, n=30 not 50, so not a strict
  delta, but the modest-signal story holds and improves on corrected labels.)
- **Prompt's first real number: 0.534**, above baseline, below Exec. The judge is a
  real but weaker standalone signal at N=5; it catches some Exec misses (`Mbpp/785`)
  and adds its own false positives (`Mbpp/252` correct → prompt=1.000).
- **Ensemble story, not "Exec alone."** No single method dominates every case; the
  three flag different incorrect mains. Improvement-2 narrative = complementary
  signals, with Exec as the anchor.

## AST-metric recommendation (feeds iteration 4)

**Keep bag-of-node-types Jaccard as the AST metric for now.** It discriminates above
baseline with clean low-end specificity and zero dependency/cost. A structure-sensitive
metric (tree edit distance, subtree n-grams) is **not** justified by this readout — the
MVP metric is informative, not degenerate.

The higher-value iteration-4 target is **AST false positives** (finding 5), not metric
sophistication: correct-but-structurally-diverse samples inflate the score. This is the
same false-positive weakness named for Exec; address them together (reference-vs-sample
normalization), and only revisit a richer metric if false-positive hardening alone
leaves AST behind Exec.
