---
type: plan
status: ready
created: 2026-06-12
source_plan: "[[01-codecheck-roadmap]]"
---

# Iteration 6: Improvement-2 analysis + report (MBPP+ only)

## Context

Improvement 2 (code-hallucination detection) is implemented across four methods
(exec / prompt / ast / code_bert) and three datasets. This iteration produces the **analysis
and the written report**. Scope is **MBPP+ only** for now (HumanEval+ and CodeHaluEval are not
ready to analyze); both available models are in scope as the cross-model comparison:

- `results/codecheck-mbpp-300-gemma4-31B.json` — n=300, incorrect=75, baseline 0.250
- `results/codecheck-mbpp-300-qwen3.5-9B.json` — n=300, incorrect=86, baseline 0.287

The report mirrors the KAIST slide deck `docs/source/03-project-progress.pdf` (gray section
label -> bold title -> column cards / tables / PR-curve grids -> "Key Conclusion" callout) and
the replication's AUC-PR table layout. **Every method's AUC-PR is split into detect-incorrect
and detect-correct, read against the prevalence baseline.**

## The numbers this report presents (already computed, MBPP+ 300)

AUC-PR detect-incorrect / detect-correct (baseline = incorrect prevalence):

| Method | gemma4-31B inc | gemma cor | qwen3.5-9B inc | qwen cor |
|--------|:---:|:---:|:---:|:---:|
| **exec** | **0.733** | 0.930 | **0.728** | 0.926 |
| **prompt** | 0.584 | 0.912 | 0.700 | 0.903 |
| **ast** | 0.414 | 0.885 | 0.598 | 0.886 |
| **code_bert** | 0.371 *(n=100)* | 0.780 | 0.454 | 0.841 |
| *baseline* | *0.250* | — | *0.287* | — |

Correlation (Spearman, score vs incorrect): exec 0.59/0.55, prompt 0.44/0.48, ast 0.31/0.42,
code_bert 0.18/0.28.

**Headline story:** exec > prompt > ast > code_bert on detect-incorrect, both models.
code_bert sits at/near the baseline (≈ no skill) — the negative result. detect-correct is
uniformly high but **inflated by the 75% correct prevalence**, so it must be read against the
baseline, not in isolation. Spearman ≥ Pearson for exec (a rank signal with tie pile-ups at 0).

## Deliverables

- **`docs/analysis/report.md`** — the report, slide-deck-style sections (below).
- **`docs/analysis/images/*.png`** — generated figures.
- **`scripts/make_analysis_figures.py`** — committed, regenerable matplotlib generator that
  reads the two MBPP+ result files and writes every PNG. Reuses the exact label/score
  construction from `codecheck/evaluate.py` (`y_true = 0 if is_correct else 1`; detect-correct
  negates the score) and sklearn `precision_recall_curve` + trapezoidal `auc`, so figure AUCs
  match the `evaluate` command's numbers.

## Visualizations (the chosen set, each justified)

1. **AUC-PR bar chart (headline).** Grouped bars: 4 methods × {detect-incorrect,
   detect-correct}, dashed baseline line per model, faceted by model (2 facets). One figure.
   *Why:* the single slide that states the whole result and the method ranking against the
   no-skill floor.
2. **PR-curve grid (per model).** 2 panels (detect-incorrect, detect-correct), 4 method curves
   + random baseline, trapezoidal AUC annotated in the legend. One figure per model (2 figures).
   *Why:* the deck's Figure-5 style; shows *where* on the recall axis each method wins, not
   just the scalar.
3. **Per-class score histograms.** Per model, 4 subplots (one per method), stacked
   correct/incorrect counts across 10 score bins. One figure per model (2 figures).
   *Why:* exposes the tie pile-up at score≈0 that makes the scalar AUC fragile and explains the
   confident-consistent misses (incorrect mains sitting in the [0.0,0.1) bin).
4. **Confident-consistent blind-spot illustration.** A qualitative callout: one real MBPP+
   incorrect main whose exec≈0 (samples agree with the wrong main), shown as
   main-code + a sample + the count breakdown. *Why:* makes the blind spot concrete; it is the
   bridge to Improvement 1's high-certainty-hallucination theme. (The plan's build step picks a
   specific task_id from the [0.0,0.1)-bin incorrect set; ~46 such cases exist for gemma.)

Dropped as low-value at this scope: a method×dataset ranking heatmap (only one dataset now —
the table suffices); a cross-dataset panel (out of scope until the other datasets are ready).

## Report structure (`docs/analysis/report.md`, slide-deck sections)

1. **Improvement 2 — what & why.** Code hallucination = incorrect code; consistency across
   resamples as the zero-resource signal; the four methods in one line each.
2. **Setup.** MBPP+ 300, two local models (gemma4-31B, qwen3.5-9B), n=20 samples, exec ground
   truth via the canonical, four scorers on identical generations.
3. **Result — AUC-PR table** (the table above) + the bar-chart figure.
4. **Result — PR curves** (the two grid figures) + reading them against the baseline.
5. **Per-method analysis.** exec (strongest), prompt (second; LLM-judge), ast (weak structural
   signal), code_bert (negative result — saturated, ≈ baseline) with the histogram figures.
6. **The confident-consistent blind spot** (illustration) + link to Improvement 1.
7. **Key Conclusion** callout — exec is the reliable code-hallucination detector; behavioral
   consistency >> structural/embedding similarity; the blind spot is the shared limit.

## Caveats to surface in the report

- **code_bert coverage:** gemma4 code_bert is **n=100, not 300** (partial offline pass) — label
  it on every code_bert figure/number; qwen is full n=300.
- **detect-correct inflation:** high AUC is partly the 75% correct prevalence; always shown next
  to the baseline.
- **code_bert saturation:** anisotropic embeddings, cosine≈1 for all code — the documented
  negative result, not a bug.
- **confident-consistent blind spot:** when the model is consistently wrong, no consistency
  method separates it without an oracle.
- **scope:** MBPP+ only; HumanEval+/CodeHaluEval analysis deferred (noted as future work).

## Critical files

- `scripts/make_analysis_figures.py` (new), `docs/analysis/report.md` (new),
  `docs/analysis/images/` (new dir).
- Reuse: `codecheck/evaluate.py` primitives; `codecheck/pipeline.load_results`;
  `codecheck/models.CodeResult`.

## Verification

1. `python scripts/make_analysis_figures.py` regenerates all PNGs into `docs/analysis/images/`,
   no error, on both result files.
2. Each figure's annotated AUC matches `run_codecheck.py evaluate` for the same file/method
   (sanity: exec gemma detect-incorrect ≈ 0.733).
3. Report renders (markdown), every figure referenced resolves, every number traces to an
   `evaluate` readout or the figure script.
4. Commit (no push), per project convention.

## Open notes

- Figures are static PNG (matplotlib), committed + regenerable — no notebook.
- When HumanEval+/CodeHaluEval are ready, this report extends with the cross-dataset panel and
  the dataset-comparison table; structure already leaves room.
