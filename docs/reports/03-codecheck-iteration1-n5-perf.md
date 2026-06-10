# CLI User Test — N=5 Run + Performance Fixes (Iteration 1)

Date: 2026-06-10
Target: `python run_codecheck.py run --limit 5 --n 5 --seed 0` / `evaluate`
Follows reports 01 (offline) and 02 (first live run). This round did a real N=5 run,
then root-caused and fixed the two latency bottlenecks it exposed.
Artifact: `03-codecheck-iteration1-n5-result.json` (the raw results).

## N=5 results (limit 5, seed 0)

| Problem | correct | exec_score |
|---------|---------|-----------|
| Mbpp/435 | True | 0.343 |
| Mbpp/459 | True | 0.442 |
| Mbpp/62  | True | 0.000 |
| Mbpp/270 | True | 0.000 |
| Mbpp/591 | True | 0.000 |

`evaluate` → **AUC-PR = nan, n=5, all 5 correct.**

## 🔑 Finding: "MBPP+ too smart" risk confirmed (preliminary)

All 5 main implementations (T=0) were correct → single class → AUC-PR undefined (`nan`).
The roadmap's central gating question — *do recent models hallucinate enough on MBPP+ to
give signal?* — looks like **no** on this easy, low-numbered slice. `exec_score` still
varies (0.0–0.44: samples diverge on negative-input edge cases even when main is correct),
but with no incorrect mains there is nothing to detect. Points toward pulling the
hallucination-targeted datasets (CodeHaluEval / Collu-Bench) earlier, a weaker generation
model, or a harder problem slice — needed before AUC-PR is meaningful.

## 🔴→🛠 Performance: two bottlenecks found and fixed

The first N=5 run took **7m40s** for 5 problems. Root-caused in two stages:

**1. Generation — reasoning tokens (fixed).**
`qwen/qwen3.5-9b` is a reasoning model. Per trivial function it emitted thousands of hidden
chain-of-thought tokens (measured: 4199 reasoning tokens, 83.5s for one call) before a
3-line answer that we keep. The same prompt on the OpenRouter web UI returned in ~2s
(reasoning off there). Fix: pass `reasoning.enabled=false` on the generation calls →
**83.5s → 3.7s per call (~22×)**, identical code. (`max_tokens` would be wrong: reasoning
precedes the answer, so a cap truncates mid-think.) Commit `89cc53b`.

**2. Execution — spawn-per-input (fixed).**
With generation fast, execution dominated: each problem ran main + N samples + canonical =
7 impls × ~106 inputs, **one spawned process per (impl, input)** ≈ 742 spawns/problem.
Fix: batch all inputs for an implementation into **one** spawn, with a per-input `SIGALRM`
timeout so a single hanging input still degrades to `timeout` without blocking the rest
(hard hang → overall deadline → process killed). Measured on Mbpp/435 (106 inputs, n=5):
**45.2s → 0.4s (112×)**, identical scores. Commit `d21eaab`.

Net: a 5-problem N=5 run drops from ~7m40s to ~2 min (now generation-bound at ~3.7s/call,
execution negligible).

## Status

- Unit suite: 30 passing (added batch-execution + reasoning-off tests).
- Live path works and is now fast enough to scale N and `--limit`.
- Open: AUC-PR is `nan` on easy MBPP+ — need an incorrect class. Next test should use a
  harder slice or a weaker model to confirm the signal separates correct from incorrect.
