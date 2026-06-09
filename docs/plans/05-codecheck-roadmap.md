---
type: plan
status: draft
created: 2026-06-10
source_plan: "[[04-codecheck-methods]]"
---

# Iteration Roadmap — SelfCheckGPT for Code

## Context

This roadmap sequences the implementation of the three code-domain SelfCheck
variants defined in `04-codecheck-methods.md` (SelfCheck-Exec, SelfCheck-AST,
SelfCheck-Prompt). It is a macro-sequence: each iteration ships something
runnable end-to-end on real coding problems and produces a real
consistency-vs-correctness readout the team can judge.

**Confirmed sequencing decisions (forks resolved with the team):**
- **MVP method = SelfCheck-Exec.** It carries the most code-native signal
  (behavioral I/O consistency) and forces the execution harness early — that
  harness doubles as the ground-truth correctness labeler, so it is needed
  regardless.
- **First dataset = MBPP+.** Rich executable test suites (EvalPlus) make it the
  cleanest surface for an execution harness and ground-truth labeling.
- **Generation pipeline built fresh** (not reused from the WikiBio replication):
  code tasks need their own prompting and implementation extraction.

**Headline value:** detect hallucinated (incorrect) code implementations using
only sample-consistency — no human annotation — extending SelfCheckGPT to the
coding domain.

**The loop:** after each iteration the team runs it directly and judges the
output. That feedback revises the iterations not yet detailed below — when we
return to detail iteration N+1, we fold in what was learned rather than
executing the stale outline. The biggest open question feedback must answer
early: *do recent models actually hallucinate enough on MBPP+ to give signal?*
(Same "models too smart" risk flagged for GSM8K in the progress doc.)

## Overview

| # | Iteration | User-facing slice |
|---|-----------|-------------------|
| 1 | Exec MVP on MBPP+ | Run a command on a few MBPP+ problems; see each implementation's behavioral-consistency score next to its true correctness, plus a first AUC-PR. |
| 2 | Prompt variant | Score the same data with the LLM-judge variant; compare Exec vs Prompt AUC-PR side by side. |
| 3 | AST variant | Score the same data with AST structural similarity; three-way compare Exec / Prompt / AST. |
| 4 | Scale on MBPP+ | Full N=20 and full problem set; paper-comparable AUC-PR + correlation. *(light — re-planned from feedback)* |
| 5 | Add hallucination datasets | Run all three variants on CodeHaluEval, then Collu-Bench. *(light)* |
| 6 | Analysis + report | Synthesize across variants and datasets into the Improvement-2 narrative. *(light)* |

---

## Iteration 1 — SelfCheck-Exec MVP on MBPP+

- **Goal:** prove that behavioral-consistency across sampled implementations
  separates correct code from hallucinated code, and stand up the shared
  generation + execution harness that every later iteration depends on.
- **User-facing value:** run one command on ~10–20 MBPP+ problems and see, per
  implementation, its Exec consistency score next to its true correctness (from
  the problem's tests), plus a first AUC-PR number — enough to eyeball whether
  the signal works.
- **Features introduced:**
  - Fresh code-generation pipeline: 1 main implementation at `T=0`, `N` samples
    at `T=1` per problem (start small, e.g. `N=5`).
  - MBPP+ loader (problems + reference tests).
  - Execution harness: run any implementation on a shared input set, capture
    output / exception / timeout in a sandbox.
  - Ground-truth labeler: mark each implementation correct/incorrect by running
    the problem's reference tests.
  - SelfCheck-Exec scorer: compare outputs of main vs each sample on the shared
    inputs → per-implementation consistency score.
  - AUC-PR readout against the ground-truth labels.
- **Deliverables:**
  - Runnable command (e.g. `python -m codecheck.run --method exec --dataset mbpp_plus --n 5 --limit 20`).
  - Output table / JSON: `(problem_id, impl_id, exec_score, is_correct)`.
  - Printed AUC-PR for the subset.
- **Testable conditions:**
  - Pipeline produces 1 main + `N` samples for each selected problem.
  - Harness executes every implementation on the shared inputs and records a
    result for each, including crashes and timeouts (no hang).
  - Each implementation gets a ground-truth correct/incorrect label from tests.
  - Exec score computed for every implementation.
  - AUC-PR prints without error on the subset.
- **User test flow:** run the command on a 10–20 problem subset; open the table;
  scan whether low-consistency rows line up with incorrect implementations; read
  the AUC-PR.
- **Feedback to collect:**
  - Does Exec consistency visibly separate correct from incorrect? (go/no-go for
    the whole code direction)
  - **Hallucination rate:** are recent models too good on MBPP+ to produce
    enough incorrect implementations? (decides whether iteration 5 must come
    sooner)
  - Right input set to compare on — reference test inputs vs. generated/fuzzed
    inputs?
  - Output-comparison rule edge cases: floats/tolerance, exceptions, unordered
    collections, stdout vs return value.
  - Sandbox needs (timeout value, resource limits, isolation) before scaling.
  - Sensible `N`.
- **Risks / open decisions:**
  - Sandbox safety — executing model-generated code requires isolation; settle
    the minimum viable sandbox here.
  - "Models too smart" on MBPP+ → low hallucination rate could starve the
    signal; if so, reorder iteration 5 (hallucination-targeted datasets) earlier.

---

## Iteration 2 — SelfCheck-Prompt variant

- **Goal:** add the LLM-as-judge consistency signal and get the first
  head-to-head comparison against Exec on identical data.
- **User-facing value:** re-score the same MBPP+ subset with `--method prompt`
  and see Exec vs Prompt AUC-PR side by side, to judge which signal is stronger.
- **Features introduced:**
  - Prompt scorer using the template from `04-codecheck-methods.md` (judge LLM
    compares each sample unit against the main implementation).
  - Yes / No / N-A aggregation across the `N` samples (mapping per the original
    Prompt variant, e.g. Yes→0.0 / No→1.0 / N-A→0.5).
- **Deliverables:**
  - Prompt scorer module wired into the same run command.
  - Comparison table / readout: Exec vs Prompt AUC-PR on the same subset.
- **Testable conditions:**
  - Judge returns a parseable Yes/No/N-A for each (unit, sample) pair; parse
    failures handled, not crashing.
  - Aggregation produces a per-implementation score.
  - AUC-PR computed and shown next to Exec's on the same data.
- **User test flow:** run with `--method prompt` on the same subset; compare the
  two AUC-PR numbers and a few disagreeing rows.
- **Feedback to collect:** which signal is stronger and where they disagree;
  judge cost/latency (matters at full scale); judge parse-failure rate; whether
  the judge prompt needs tuning for code.
- **Risks / open decisions:** judge cost at full `N=20` × full dataset (feeds
  iteration 4 sizing).

---

## Iteration 3 — SelfCheck-AST variant

- **Goal:** add the structural-consistency signal and complete the three-variant
  set on one dataset.
- **User-facing value:** re-score the same subset with `--method ast` and get a
  three-way Exec / Prompt / AST comparison — does syntactic structure add signal
  beyond behavior?
- **Features introduced:**
  - AST parser for generated implementations (parse-failure handling for
    malformed samples).
  - Tree-similarity metric main vs each sample (e.g. tree edit distance or
    subtree overlap), with normalization for identifier renaming.
- **Deliverables:** AST scorer module; three-way comparison table.
- **Testable conditions:** malformed samples handled without crashing; similarity
  computed per implementation; AUC-PR computed; three variants shown together.
- **User test flow:** run `--method ast` on the same subset; read the three-way
  table.
- **Feedback to collect:** does structure add signal beyond Exec/Prompt? which
  AST metric behaves best? interaction with formatting-only differences.
- **Risks / open decisions:** AST-metric choice may need its own small
  experiment — defer detail until this iteration starts.

---

## Iteration 4 — Scale on MBPP+ *(light — re-planned from feedback)*

- **Goal:** produce paper-comparable numbers on the full MBPP+ set at full
  `N=20`, with AUC-PR computed consistently with the replication (settle
  trapezoidal-AUC vs average-precision) plus aggregate correlation.
- **User-facing value:** a full results table for all three variants on MBPP+.

## Iteration 5 — Add hallucination-targeted datasets *(light)*

- **Goal:** run all three variants on CodeHaluEval, then Collu-Bench.
- **User-facing value:** cross-dataset results; directly addresses the
  "models too smart on MBPP+" risk by using datasets built to surface code
  hallucination. *May be pulled earlier if iteration 1 shows MBPP+ hallucination
  rate is too low.*

## Iteration 6 — Analysis + report *(light)*

- **Goal:** synthesize Exec vs AST vs Prompt across datasets into the
  Improvement-2 narrative for the final report.
- **User-facing value:** the written comparison + conclusions.
