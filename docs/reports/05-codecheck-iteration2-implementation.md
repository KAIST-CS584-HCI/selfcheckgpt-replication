# Iteration 2 — SelfCheck-Prompt: Implementation Report

Date: 2026-06-10
Branch: `wonseok/feat/codecheck-prompt` (off main `bb17dbc`; local, not pushed)
Plan: `docs/plans/03-codecheck-iteration2-prompt.md`
Method: subagent-driven TDD (implementer + spec review + quality review per task, plus a
final integration review). 13 commits, 56 unit tests passing.

## Goal

Add the LLM-as-judge consistency variant (SelfCheck-Prompt) alongside SelfCheck-Exec, and
score the same generated implementations with both, so Exec vs Prompt AUC-PR can be compared
head-to-head on identical MBPP+ data. The driving question: **does the judge catch the
confident-consistent hallucinations Exec scores ≈0 on** (Exec's named blind spot from
iteration 1)?

## Features introduced

- **Per-method scoring model.** `CodeResult` now carries `scores: dict[str, float]` (method
  name → inconsistency score, higher = more likely incorrect) instead of a single
  `exec_score`. Back-compatible: old iteration-1 result JSON (bare `exec_score` key) still
  loads and evaluates. `exec_score` kept as a read-only property.
- **`n_inputs` on every result.** Records K, the size of the shared input set each problem's
  implementations were run on (recorded for the record; see iteration-1 input-set findings).
- **SelfCheck-Prompt scorer** (`codecheck/prompt_score.py`):
  - `PromptJudge` — for each sample, asks the judge LLM whether the sample's behavior is
    consistent with the main implementation; fires the N calls concurrently
    (`ThreadPoolExecutor`), reasoning disabled by default (fast), and accumulates a
    `parse_failures` count.
  - `parse_judgment` — maps the judge's free text to a score: Yes (consistent) → 0.0,
    No (inconsistent) → 1.0, N/A → 0.5; unparseable → 0.5 and counted as a parse failure.
  - Aggregation: mean inconsistency over the N samples — same `[0,1]` scale and direction
    as Exec.
- **Multi-method pipeline.** One run generates each problem's implementations once and
  scores them with every selected method. Correctness labeling (canonical-test execution)
  is always done. Sample execution is skipped when `exec` is not requested (prompt-only
  runs don't pay the execution cost).
- **`run --method {exec,prompt,both}`.** Selects the scorer(s). The judge reuses
  `OPENROUTER_MODEL`. After a judged run the CLI prints the judge parse-failure count.
- **Per-method evaluation readout.** `evaluate` now prints, for every method present in the
  results file: the trapezoidal **AUC-PR**, the shared **prevalence baseline** (PR no-skill
  floor), and a **per-class score histogram** (10 bins, correct/incorrect counts). Tolerant
  of mixed-method files. This is the Exec-vs-Prompt side-by-side the iteration exists to
  produce, with the rigor items iteration 1 asked for.

## Deliverables

- `codecheck/prompt_score.py` (new) — judge template, parser, `PromptJudge`.
- `codecheck/models.py` — `scores` dict + `n_inputs`.
- `codecheck/evaluate.py` — `auc_pr_detect_incorrect(results, method=)`, `prevalence_baseline`,
  `score_histogram`.
- `codecheck/pipeline.py` — `score_problem`/`run_dataset` take `methods` + `judge`.
- `run_codecheck.py` — `--method` flag, judge wiring, `format_evaluation` readout.
- `codecheck/README.md` — `--method` usage + standing run/report protocol.
- Tests: `test_codecheck_prompt_score.py` (new) + updates to models/evaluate/pipeline/cli.

## Testable conditions (all covered by unit tests, 56 passing)

- Judge returns a parseable Yes/No/N-A per (main, sample) pair; parse failures handled and
  counted, not crashing.
- Aggregation produces a per-implementation prompt score; AUC-PR computed via the same
  `evaluate` function as Exec (apples-to-apples).
- One run generates + executes + labels once; exec and prompt scores share identical codes.
- Prompt-only runs skip sample execution.
- `evaluate` prints AUC-PR + baseline + histogram per method; mixed-method files don't crash.
- Back-compat: legacy `exec_score`-keyed JSON loads and evaluates.

## User test flow (live — needs `OPENROUTER_API_KEY`)

Not yet run live; these are the steps to exercise it.

1. **Run both methods on a proper random sample** (standing protocol: random full-set,
   capable model — not the easy low-numbered slice):
   ```bash
   python run_codecheck.py run --method both --limit 30 --n 5 --seed 1 --timeout 5 \
     --output output/iter2-both.json
   ```
   Watch the printed `Judge parse failures: N` at the end.

2. **Read the comparison:**
   ```bash
   python run_codecheck.py evaluate --results output/iter2-both.json
   ```
   You get, per method: AUC-PR, the shared prevalence baseline, and the per-class histogram.

3. **What to look for (the iteration's core question):**
   - Read each AUC-PR **against the baseline**, not in isolation.
   - In the histograms, find incorrect mains that Exec puts in bin `[0.0,0.1)` (its
     confident-consistent misses). Do those same problems get a **high `prompt` score**?
     That is the signal the judge complements Exec.
   - Compare the two AUC-PR numbers and scan a few rows where the methods disagree.

4. **Sanity / robustness (optional):**
   ```bash
   python run_codecheck.py run --method prompt --limit 5 --n 3        # prompt-only, no exec cost
   python run_codecheck.py evaluate --results /tmp/nope.json          # clean error, exit 1
   ```

## Feedback to collect (feeds later iterations)

- Does Prompt catch Exec's confident-consistent misses? → decides whether the Improvement-2
  story is "Exec alone" or "Exec + judge ensemble."
- Judge parse-failure rate and cost/latency at this N → feeds iteration-4 sizing and whether
  the judge is affordable on the full set.
- Whether the judge prompt needs code-specific tuning.

## Open / deferred

- **Live run pending** (user-side API key).
- **Unit granularity** (per-problem main-impl label vs per-sample) remains OPEN in the
  methods doc — affects all variants; not decided here.
- Exec input-set false-positive hardening is deferred to roadmap iteration 4.

Part of the iteration roadmap (`docs/plans/01-codecheck-roadmap.md`).
