---
type: plan
status: approved
created: 2026-06-10
source: brainstorming session (TED enhancement for SelfCheck-AST)
---

# SelfCheck-AST — Tree Edit Distance metric

## Goal

Add a structure-aware AST metric (tree edit distance) alongside the current
bag-of-node-types Jaccard, selectable via `--ast-metric {jaccard,ted}` with **`ted`
as the default**. The Jaccard metric is count-based: it ignores nesting and ordering,
so two programs with the same node-type counts but different structure score
identical, and longer-but-correct samples inflate the score. TED compares the actual
tree shapes, capturing the structural divergence Jaccard throws away. Coexistence lets
us run a Jaccard-vs-TED head-to-head on the same data and prove TED helps before
relying on it.

## Decisions (from brainstorming)

- **Implementation:** `zss` library (pure-Python Zhang-Shasha). Less code to own than
  hand-rolling; battle-tested. New dependency, added to `requirements.txt`.
- **Coexist, selectable.** Keep Jaccard; add TED as `--ast-metric`. Default `ted`.
- **No large-tree cap.** Generated MBPP+ functions are small; run TED regardless.
- **Aggregation unchanged:** mean per-sample dissimilarity over the N samples (the
  best-performing aggregation measured on the iter-3 sample, rank-AUC 0.778).

## Architecture

`codecheck/ast_score.py` gains a TED path beside the existing Jaccard one:

- **Keep** `ast_fingerprint` + `ast_dissimilarity` (Jaccard path), unchanged.
- **Add** `ast_to_tree(code) -> zss.Node | None` — parse to AST, convert to a `zss`
  tree whose **node label is the AST node-type name only** (`BinOp`, `Call`, …), never
  identifier text or literal value. This preserves the rename/literal invariance the
  Jaccard metric has. Returns `None` on parse failure or a body-less module (reuses the
  existing `(SyntaxError, ValueError, TypeError)` + empty-body guard).
- **Add** `ted_dissimilarity(main_code, sample_code) -> float` —
  `zss.simple_distance` (unit insert/delete/relabel costs, label equality), normalized
  to `[0,1]` as `dist / (size_main + size_sample)`. Since relabel(1) ≤ delete(1) +
  insert(1), `dist ≤ size_main + size_sample`, so the ratio is bounded. Both-empty →
  `0.0`. Higher = more divergent (same scale/direction as Jaccard, Exec, Prompt).
- **`ASTScorer(metric="ted")`** — constructor takes the metric; `evaluate()` dispatches
  to the Jaccard or TED per-sample function. `parse_failures` accounting and
  mean-over-samples aggregation unchanged. Unparseable sample/main → `1.0` and counted.

No change to `pipeline.py`: the metric is baked into the `ASTScorer` instance the CLI
constructs and passes through.

## Data flow / CLI

- `run --ast-metric {jaccard,ted}`, default `ted`. Meaningful only when `ast` is in the
  method set (`--method ast` or `all`); ignored otherwise.
- `run_codecheck.py` builds `ASTScorer(metric=args.ast_metric)`. The run header prints
  the active AST metric.
- `requirements.txt` gains `zss`.
- `codecheck/README.md` documents `--ast-metric` and notes TED is the default.

## Testing

- TED unit tests mirror Jaccard: rename-invariant → 0.0, literal-invariant → 0.0,
  identical → 0.0, structurally different → (0, 1], normalization stays ≤ 1.0, parse
  failure → 1.0 and counted.
- **Discrimination test (the point of the metric):** a hand-built pair with the *same
  node-type counts* but *different nesting* — Jaccard ≈ 0 while TED > 0. Locks in that
  TED sees structure Jaccard cannot.
- `ASTScorer` dispatch: both metrics produce a score; `metric="ted"` is the default.
- CLI: parses `--ast-metric`, default `ted`; unknown value rejected.

## Validation deliverable

One run comparing both metrics on the same `--no-random` sample currently in use, then
a short report (`docs/reports/09-codecheck-ast-ted-result.md`): per-class means and
rank-AUC for jaccard vs ted, answering whether TED separates correct from incorrect
better than Jaccard on this data. Prototype baseline: Jaccard ≈ 0.78 rank-AUC on the
seed-1 sample; TED is the real test of whether structure-awareness lifts it.

## Risks / open

- TED cost is higher than Jaccard (~O(n²) per pair). At N samples × problems this is
  more CPU, but runs are API-bound, so wall-clock impact is minor. No cap by decision.
- If TED does **not** beat Jaccard on the validation run, that is itself a finding
  (structure tracks correctness only weakly on MBPP+ — the iteration hypothesis), and
  Jaccard stays available. The metric switch makes that comparison cheap to repeat.
