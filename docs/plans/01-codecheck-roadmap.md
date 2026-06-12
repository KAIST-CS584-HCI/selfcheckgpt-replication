---
type: plan
status: draft
created: 2026-06-10
revised: 2026-06-12
source_plan: "[[04-codecheck-methods]]"
---

# Iteration Roadmap — SelfCheckGPT for Code

## Context

This roadmap sequences the three code-domain SelfCheck variants defined in
`04-codecheck-methods.md` (SelfCheck-Exec, SelfCheck-AST, SelfCheck-Prompt).
It is a macro-sequence: each iteration ships something runnable end-to-end on
real coding problems and produces a real consistency-vs-correctness readout the
team can judge.

**The loop:** after each iteration the team runs it directly and judges the
output. That feedback revises the iterations not yet detailed below — when we
return to detail iteration N+1, we fold in what was learned rather than
executing the stale outline.

**Confirmed sequencing decisions:**
- **MVP method = SelfCheck-Exec.** Most code-native signal (behavioral I/O
  consistency); its execution harness doubles as the ground-truth correctness
  labeler, so it is needed regardless.
- **First dataset = MBPP+.** Rich EvalPlus test suites give clean executable
  ground truth.
- **Generation pipeline built fresh** (not reused from WikiBio): code tasks need
  their own prompting and implementation extraction.

### Revision (2026-06-10) — folding in iteration-1 evidence

Iteration 1 ran live on MBPP+. Two confirm experiments (random 50 problems,
capable model `qwen3.5-9b`; and 30 problems, weak model `llama-3.2-3b`) settled
the roadmap's central open questions and overturn two earlier assumptions:

- **"Models too smart on MBPP+" — false alarm.** The all-correct `nan` result
  came from the *easy low-numbered seed-0 slice*, not the dataset. A **random
  full-set sample with a capable model gives 28% incorrect mains** — a real
  positive class. **MBPP+ stays the primary dataset; hallucination-targeted
  datasets are NOT pulled early.**
- **Weak-model lever — rejected.** A weak model produces more incorrect mains
  (56%) but **degenerate `exec_score`** (29/30 = 0.0): consistently-wrong code →
  samples agree with the wrong main → zero divergence → undetectable. Its
  AUC-PR 0.796 was an artifact of single-point interpolation. Worse than the
  capable model, not better.
- **Real Exec readout (capable, random 50):** AUC-PR **0.4545** vs a prevalence
  floor of **0.28** — real but modest lift. Of 14 incorrect mains, 3 score
  exactly 0.0 (undetectable) and most score low; only 3 are strongly flagged.
- **Two named weaknesses to carry forward:**
  1. **False positives** — some *correct* mains score high because their samples
     diverge on EvalPlus adversarial inputs (`exec_score` measures
     sample-divergence-from-main, not main correctness). Hurts precision.
  2. **Confident-consistent blind spot** — Exec cannot see hallucinations where
     all samples agree on the same wrong code. This is the same failure mode
     **Improvement 1** targets → a cross-improvement narrative for the report.
- **Throughput — solved in iteration 1.** Reasoning-off (~22×) + batched
  execution (~112×). Scale is no longer a standalone risk; the old "scale"
  iteration is now light.
- **Standing protocol for every run from here:** random full-set sampling +
  capable model + sensible N; and every readout prints the **prevalence
  baseline** and a **per-class `exec_score` histogram** next to the scalar
  AUC-PR (trapezoidal, matching the replication), because heavy ties at 0 make
  the bare number fragile.

Reports: `docs/reports/01..03-codecheck-iteration1-*`.

### Revision 2 (2026-06-10) — before detailing iteration 3 (AST)

Iteration 2 (Prompt) was built, tested (58 unit tests), and **merged to main (#4)**
— but the feedback loop this roadmap runs on is currently **broken**, so AST is
not detailed until a gate run closes it:

- **Iter-2's defining question is unanswered.** The live `--method both` run
  (does the judge catch Exec's confident-consistent misses?) was **never done** —
  only one trivial 1-sample judge call. There is **zero iter-2 feedback** to fold
  into AST. Detailing AST now would execute a stale outline.
- **Iter-1's headline number is now suspect.** The `PYTHONHASHSEED`
  nondeterminism bug (set/dict output ordering across spawned subprocesses) was
  fixed in commit `cdc124f` **after** iter-1's live run. It silently mislabeled
  set/dict-returning problems (correct canonical code scored incorrect),
  corrupting **both** the label and the score. So **AUC-PR 0.4545 vs 0.28** was
  measured on tainted data — direction of error unknown. Found via the iter-2
  CLI user test (`docs/reports/06`).
- **Fix:** insert **Iteration 2.5 — Validation run (gate)** below. One live
  both-method run, post-fix, re-establishes a trustworthy Exec number and the
  first real Prompt number + head-to-head, in a single run. AST is detailed only
  after, folding in that feedback.
- **Unit granularity OPEN — RESOLVED.** Settled as **per-problem /
  per-main-implementation** in `04-codecheck-methods.md` (only the T=0 main impl
  is a scored unit; the N samples are evidence only). Paper-faithful; applies to
  all three variants incl. AST. Low-positive-count fragility is handled by
  scaling problem count (the manual `--limit 300` run), not by redefining the unit.
- **Exec false-positive hardening — partly a determinism artifact.** The
  hashseed bug confirmed part of iter-1's "false positives" was nondeterministic
  ordering (now fixed), the rest is genuine edge-case divergence on EvalPlus
  adversarial inputs. Iter-4 hardening should separate the two; the determinism
  half is already closed.

### Revision 3 (2026-06-12) — four variants landed; real multi-method numbers; re-sequence

Iterations 2.5 (validation gate) and 3 (AST) are **done**, an unplanned **fourth
variant — SelfCheck-CodeBERT** landed (embedding similarity), and a layer of
robustness/infra work shipped. Two ~250-problem MBPP+ runs (qwen3.5-9b, gemma4-31b)
now carry a real **four-method** readout. This revision folds that evidence in.

- **Four methods, one consistent ranking.** detect-incorrect AUC-PR, both models
  (baseline ≈0.24-0.26):

  | method | qwen3.5-9b (n=252) | gemma4-31b (n=254) |
  |--------|-----|-----|
  | exec     | **0.68** | **0.73** |
  | prompt   | 0.67 | 0.58 |
  | ast      | 0.57 | 0.39 |
  | code_bert| 0.41 | 0.33 (n=54) |

  Order **exec > prompt > ast > code_bert** holds across both. Spearman tracks it.
- **Structure/embedding underperform — hypothesis confirmed.** AST and CodeBERT,
  the two *local-similarity* signals, score weakest; CodeBERT is near-baseline on
  gemma (pearson 0.05). Consistent with the confident-consistent blind spot:
  consistently-wrong code is also structurally/embedding-similar to its wrong main,
  so divergence ≈0 and the error is missed. **CodeBERT is largely a negative
  result** on MBPP+ (embedding similarity ≠ correctness signal) — keep it for the
  "what doesn't work" half of the report, not as a headline method.
- **Prompt does NOT clearly beat Exec on MBPP+** (tied on qwen, worse on gemma). So
  iter-2's defining question ("does the judge catch Exec's blind spot?") is *not*
  answered affirmatively here — both are mid-tier. The blind spot needs a dataset
  where hallucinations are *deliberately consistent* → raises the priority of the
  targeted-dataset iteration (now iter 5), which becomes the narrative's real test.
- **AST metric settled:** jaccard default; TED scored marginally worse on MBPP+ and
  is kept available, not default (`docs/reports/09`).
- **Prompt parser audited clean.** The Yes/No/N-A regex is first-match-anywhere
  (could be stolen by the justification), but across 9,640 real judgments: 0 silent
  mis-parses, 1 empty reply. Anchored-regex hardening is *optional/defensive*, not a
  correction — demoted to a small iter-4 robustness item.
- **Execution worst-case cost.** `--timeout` is per-input; a sample that loops on
  *every* input costs ≈ n_inputs × timeout (one comb-sort sample stalled a run ~55
  min). Mitigated cheaply by lowering the default to `--timeout 2`; a per-batch
  ceiling (K-timeout) was considered and **dropped** as unneeded.
- **Infra that already shipped (enabling, not iterations):** continue-on-failure,
  incremental JSON save + task_id resume, daemon-thread API timeout (clean exit),
  parallel sample execution, torch/NaN/corrupt-file guards, and the flat→subpackage
  refactor (`score/`, `generation/`, `execution/`). Scale is produced manually via a
  `--limit 300` run; no separate full-set run-iteration is needed.

## Overview

| # | Iteration | Status / slice |
|---|-----------|----------------|
| 1 | Exec MVP on MBPP+ | ✅ done. Per-impl behavioral-consistency score + first AUC-PR. |
| 2 | Prompt variant | ✅ built + merged (#4). LLM-judge consistency on the same data. |
| 2.5 | Validation run *(gate)* | ✅ done. Post-hashseed-fix run re-established trustworthy Exec + first real Prompt numbers; head-to-head produced. |
| 3 | AST variant | ✅ done. Jaccard (default) + TED metric; three-way compare. Structure is the weak signal. |
| 3.5 | CodeBERT variant | ✅ done. Offline embedding-similarity scorer reusing saved codes (`codebert` subcommand). Weakest signal — negative result on MBPP+. |
| 5 | Second dataset (HumanEval+) | ✅ implemented. `--dataset humaneval` (loader-only, reuses pipeline + all four scorers). CodeHaluEval deferred (stdin/stdout path), Collu-Bench dropped (logit-detection benchmark). |
| 6 | Analysis + report | Four-method × dataset synthesis (MBPP+, HumanEval+); the structure/embedding negative result + blind-spot cross-link to Improvement 1. **(next)** *(light)* |
| — | *deferred* | CodeHaluEval via a stdin/stdout whole-program exec path — the real confident-consistent stress test. |

*(No iteration 4: scale is produced manually via a `--limit 300` MBPP+ run, not a
gated iteration. Leftover small/optional items are parked under "Deferred items"
below — none block iter 5/6.)*

---

## Iteration 1 — SelfCheck-Exec MVP on MBPP+

> Status: ✅ done — implemented (TDD, 30 unit tests), run live on MBPP+, findings
> folded into the revision above.

- **Goal:** prove behavioral-consistency separates correct from hallucinated code
  and stand up the shared generation + execution harness every later iteration
  depends on. **Result:** signal confirmed but modest (AUC-PR 0.45 vs 0.28);
  harness + labeler working; throughput bottlenecks fixed.
- **Delivered:** `codecheck/` package (generation, execution sandbox, exec_score,
  labeling, MBPP+ loader, pipeline, evaluate); CLI `run_codecheck.py run`/
  `evaluate`; `OPENROUTER_MODEL` env switch.
- **Key learnings:** see the revision section — "too smart" disproven,
  weak-model rejected, false-positive + confident-consistent weaknesses named,
  throughput solved, standing run/report protocol set.

---

## Iteration 2 — SelfCheck-Prompt variant

- **Goal:** add the LLM-as-judge consistency signal and get the first head-to-head
  against Exec on identical data — specifically testing whether the judge catches
  the **confident-consistent hallucinations Exec scores ≈0 on**.
- **User-facing value:** re-score the same random MBPP+ sample with
  `--method prompt` and see Exec vs Prompt AUC-PR side by side, with the
  disagreeing rows surfaced (where one flags an incorrect main the other misses).
- **Features introduced:**
  - Prompt scorer using the template from `04-codecheck-methods.md` (judge LLM
    compares each sample unit against the main implementation).
  - Yes / No / N-A aggregation across the N samples (e.g. Yes→0.0 / No→1.0 /
    N-A→0.5), producing a per-implementation score on the same scale as Exec.
  - `--method {exec,prompt}` selection on the run/evaluate path.
  - Shared readout helper that prints, for any method: AUC-PR (trapezoidal),
    **prevalence baseline**, and a **per-class score histogram** (standing
    protocol from iteration 1).
- **Deliverables:**
  - Prompt scorer module wired into the existing run command.
  - Comparison table: Exec vs Prompt AUC-PR on the *same* random sample, plus a
    short list of rows where the two disagree.
- **Testable conditions:**
  - Judge returns a parseable Yes/No/N-A per (unit, sample) pair; parse failures
    handled, not crashing, and counted.
  - Aggregation produces a per-implementation score; AUC-PR computed with the
    same `evaluate` function as Exec (apples-to-apples).
  - Readout prints baseline + histogram alongside the scalar for both methods.
  - Run uses the standing protocol (random full-set sample, capable model) — not
    the easy low-numbered slice.
- **User test flow:** run `--method prompt` on the same seeded random sample used
  for Exec; read the two AUC-PR numbers, the two baselines, and the disagreeing
  rows — focus on whether Prompt flags any of the 3 incorrect mains Exec scored
  0.0.
- **Feedback to collect:**
  - Does Prompt catch Exec's confident-consistent misses? (decides whether the
    Improvement-2 story is "Exec alone" or "Exec + judge ensemble".)
  - Judge parse-failure rate and cost/latency at this N (feeds iteration-4 sizing
    and whether judge is affordable on the full set).
  - Whether the judge prompt needs code-specific tuning.
- **Risks / open decisions:**
  - Judge cost at full N × full dataset (informed the manual `--limit 300` sizing).
  - OPEN: is the per-problem unit (label = main-impl correctness) the right
    granularity, or should samples be scored as units too? Affects all variants;
    settle in the methods doc, not here.

---

## Iteration 2.5 — Validation run (gate before AST)

> Status: ✅ done. Closed the feedback loop iterations 1 and 2 left open. Exec
> survived the label fix as the strongest signal; Prompt landed mid-tier (does not
> clearly beat Exec on MBPP+ — see Revision 3). AST unblocked and shipped.

- **Goal:** produce trustworthy numbers to slice AST against. Re-establish Exec's
  AUC-PR on corrected labels (post-hashseed-fix) and produce the first real
  Prompt AUC-PR + Exec-vs-Prompt head-to-head on identical data.
- **User-facing value:** one results file + one `evaluate` readout showing both
  methods' AUC-PR, the shared prevalence baseline, and per-class histograms — the
  comparison iter 2 was built to produce but never ran.
- **Run (standing protocol: random full-set, capable model):**
  ```bash
  python run_codecheck.py run --method both --limit 30 --n 5 --seed 1 --timeout 5 \
    --output output/iter2_5-both.json
  python run_codecheck.py evaluate --results output/iter2_5-both.json
  ```
- **Testable conditions:**
  - Run completes; `Judge parse failures: N` printed and reasonable (low).
  - Both methods scored on identical codes; AUC-PR via the shared `evaluate`.
  - No set/dict mislabeling (hashseed fix holds at scale).
- **Feedback to collect (this is what unblocks AST):**
  1. Trustworthy Exec AUC-PR vs the 0.28 baseline — does the modest-signal story
     survive the label fix?
  2. Does Prompt flag any incorrect mains Exec scores ≈0 (the blind spot)? →
     "Exec alone" vs "Exec + judge ensemble" story.
  3. Judge parse-fail rate + cost/latency at N=5 → feeds iter-4 sizing.
- **Risks / open decisions:** if Exec's corrected number collapses or the judge
  is uninformative, AST's framing ("add signal beyond Exec/Prompt") changes —
  hence this gates iter 3.

---

## Iteration 3 — SelfCheck-AST variant

> Status: ✅ done. Jaccard node-type fingerprint (default) + TED (zss) available
> via `--ast-metric`. Result: AST is the weak structural signal (qwen 0.57 / gemma
> 0.39 detect-incorrect) — hypothesis that structure shares the blind spot is
> supported. Reports `docs/reports/07..09`.

- **Goal:** add the structural-consistency signal and complete the three-variant
  set on one dataset. Direct test of the hypothesis that structure shares Exec's
  confident-consistent blind spot (consistently-wrong code tends to also be
  structurally consistent).
- **User-facing value:** re-score the same sample with `--method ast` and get a
  three-way Exec / Prompt / AST comparison — does syntactic structure add any
  signal beyond behavior and the judge?
- **Features introduced:**
  - AST parser for generated implementations, with parse-failure handling for
    malformed samples (counted, not crashing).
  - Tree-similarity metric, main vs each sample (e.g. tree edit distance or
    subtree overlap), normalized for identifier renaming so formatting-only and
    rename-only differences don't inflate divergence.
  - `--method ast` wired into the same run/evaluate/readout path.
- **Deliverables:** AST scorer module; three-way comparison table (AUC-PR +
  baseline + histogram per method on the same sample).
- **Testable conditions:**
  - Malformed samples handled without crashing; rename-only variants score as
    similar; similarity computed per implementation.
  - AUC-PR via the shared `evaluate`; three variants shown together on identical
    data.
- **User test flow:** run `--method ast` on the same seeded sample; read the
  three-way table; check whether AST flags anything Exec/Prompt miss.
- **Feedback to collect:** does structure add signal beyond Exec/Prompt? which
  AST metric behaves best? does it (as hypothesized) also miss the
  confident-consistent cases?
- **Risks / open decisions:** AST-metric choice may need its own small experiment
  when this iteration is detailed; tree-edit-distance libraries vary in cost.

---

## Iteration 3.5 — SelfCheck-CodeBERT variant *(unplanned; landed)*

> Status: ✅ done. Fourth consistency signal: `1 - cosine` of mean-pooled
> `microsoft/codebert-base` embeddings, main vs each sample. Runs **offline** over
> already-saved codes (`run_codecheck.py codebert --results <file>`) — no API,
> reuses iter-1/2/3 generations. Unlike AST it is not rename-invariant.

- **Result:** weakest of the four (qwen 0.41 / gemma 0.33 detect-incorrect,
  near-baseline; gemma pearson 0.05). Embedding similarity does not track
  correctness on MBPP+ — a **negative result**, valuable as the "what doesn't work"
  contrast in the report, not a headline method.
- **Why it matters to the narrative:** CodeBERT joins AST as a *local-similarity*
  method that shares Exec's confident-consistent blind spot, reinforcing that only
  behavior (Exec) and semantics-via-judge (Prompt) carry real signal — and even
  those are mid-tier on MBPP+, motivating iter 5.
- **Carried gap:** code_bert coverage is full on the qwen file but partial (54/254)
  on gemma; complete it so the four-way table is uniform before the report.

## Deferred items *(non-blocking — not an iteration)*

Scale is handled manually (a `--limit 300` MBPP+ run, ≈80% of 378, gives the
paper-comparable four-method table; `evaluate` already emits trapezoidal AUC-PR +
Pearson/Spearman), so there is no scale iteration. These small items remain optional
and gate nothing — pick them up if they pay off:

1. **Default `--timeout 2`** (down from 5) — cheaply bounds the all-inputs-looping
   worst case (per-batch K-timeout was considered and dropped as unneeded).
2. **Exec input-set false positives** — EvalPlus adversarial inputs inflate
   `exec_score` on correct mains whose samples differ on edge cases. Decide
   reference-vs-filtered inputs (deferred iter-1 item).
3. **Anchored Prompt parser** — pin the Yes/No/N-A regex to the verdict line
   (defensive for weaker/ramblier judges; current data parses clean → low priority).
4. **Finish gemma `code_bert` coverage** (carried from the CodeBERT variant).

## Iteration 5 — Second dataset (HumanEval+) *(implemented)*

> Status: ✅ implemented (loader + `--dataset` dispatch, tests, small-subset verified).
> The full HumanEval+ sweep is run manually, like MBPP+.

- **What shipped:** `load_human_eval_plus` (evalplus, function-call interface identical
  to MBPP+; HumanEval+ ships `canonical_solution` as the body only, so the loader
  assembles `prompt + body` to make it runnable) + `run --dataset {mbpp,humaneval}`.
  The whole pipeline and all four scorers are reused unchanged. Plan
  `docs/plans/08-codecheck-iteration5-humaneval.md`.
- **Why HumanEval+ instead of the originally-named datasets** — dataset inspection
  overturned the plan:
  - **CodeHaluEval** (`Yuchen111/CodeHaluEval`) is ~98% Codeforces **stdin/stdout**
    programs; the function-call `fn(*args)` harness can't run it. **Deferred** to a
    future iteration that adds a whole-program (stdin→stdout) exec path.
  - **Collu-Bench** (`lt-asset/collu-bench`) is a token-logprob hallucination-*detection*
    benchmark (pre-generated outputs + token annotations), **not** a generate-and-sample
    task source. **Dropped** from the roadmap.
  - HumanEval+ gives a clean second function-call dataset now (loader-only). It is **not**
    hallucination-targeted, so the deliberate confident-consistent stress test stays on
    the deferred CodeHaluEval work.
- **User-facing value:** a second cross-dataset four-method table (MBPP+ vs HumanEval+);
  does the exec > prompt > ast > code_bert ranking hold on a different problem set?
- **Deferred follow-up:** CodeHaluEval stdin/stdout exec path — the real
  confident-consistent stress test, when the whole-program harness is built.

## Iteration 6 — Analysis + report *(next — light)*

- **Goal:** synthesize Exec vs Prompt vs AST vs CodeBERT across datasets (MBPP+ and
  HumanEval+) into the Improvement-2 narrative for the final report.
- **User-facing value:** the written comparison + conclusions, with two threads:
  (a) the structure/embedding **negative result** (AST + CodeBERT add little on
  MBPP+ because they share the blind spot); (b) the cross-link — Exec's
  confident-consistent blind spot is the same high-confidence failure mode
  **Improvement 1** investigates ([[project-overview]]).
