# Confirm Experiments + Roadmap Re-slice (Iteration 1)

Date: 2026-06-10
Follows reports 01–03. Two cheap runs to settle the roadmap's central gating
questions on real data before re-slicing iterations 2–6. Artifacts:
`04-codecheck-iteration1-confirm-capable-n50.json`,
`04-codecheck-iteration1-confirm-weak-n30.json`.

## Setup

| Run | Model | Slice | N | Purpose |
|-----|-------|-------|---|---------|
| A | `qwen/qwen3.5-9b` (capable) | random 50, seed 1 | 5 | Is MBPP+ genuinely "too smart", or was the earlier all-correct result a slice artifact? |
| B | `meta-llama/llama-3.2-3b-instruct` (weak) | random 30, seed 1 | 5 | Does a weaker generator inject a usable both-class signal? |

Model switched via `OPENROUTER_MODEL`.

## Results

**Run A (capable, 50):** 14/50 incorrect mains (**28%**). `exec_score` spread is
rich — 24 distinct values, 0.0 → 0.994. **AUC-PR 0.4545** vs prevalence floor
**0.28**. Per-class: correct mean 0.156 / median 0.005; incorrect mean 0.235 /
median 0.077. Of 14 incorrect: 3 score exactly 0.0 (undetectable), most score
low, only 3 strongly flagged (0.62, 0.87, 0.99).

**Run B (weak, 30):** 17/30 incorrect (56%) but **degenerate `exec_score`: 29/30
= 0.0**, one 0.4. AUC-PR 0.796 — almost entirely single-point interpolation, an
artifact.

## Findings

1. **"Models too smart on MBPP+" — false alarm.** The all-correct `nan` (report
   03) came from the *easy low-numbered seed-0 slice*. A random full-set sample
   with a capable model produces a real 28% incorrect class. **MBPP+ stays
   primary; hallucination-targeted datasets are NOT pulled early.**
2. **Weak-model lever — rejected.** More incorrect mains but no signal:
   consistently-wrong code → samples agree with the wrong main → `exec_score` ≈ 0
   → undetectable. Worse than the capable model.
3. **Real Exec signal is modest:** 0.45 vs 0.28 floor. Two weaknesses named:
   - **False positives** — correct mains scoring high when samples diverge on
     EvalPlus adversarial inputs (`exec_score` measures sample-divergence-from-
     main, not main correctness).
   - **Confident-consistent blind spot** — Exec can't see hallucinations where
     all samples agree on the same wrong code. Same failure mode Improvement 1
     targets.
4. **Reporting rigor:** heavy ties at `exec_score=0` make the bare AUC-PR
   fragile (few distinct thresholds → mostly interpolation). Every readout should
   print the **prevalence baseline** and a **per-class score histogram** next to
   the trapezoidal scalar.

## Roadmap impact

Re-sliced in `docs/plans/01-codecheck-roadmap.md`:
- iter 2 = Prompt variant — test whether the LLM judge catches Exec's
  confident-consistent misses.
- iter 3 = AST (kept full).
- iter 4 = scale + Exec input-set hardening (fold in the false-positive fix).
- iter 5 = hallucination datasets, reframed to stress the blind spot.
- iter 6 = report, cross-linked to Improvement 1.

Standing run protocol set: random full-set sampling + capable model; baseline +
histogram printed with every AUC-PR.
