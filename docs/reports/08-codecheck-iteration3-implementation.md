# Iteration 3 — SelfCheck-AST: Implementation Report

Date: 2026-06-10
Branch: `wonseok/feat/codecheck-ast` (off main `6be1009`; local, not pushed)
Plan: `docs/plans/04-codecheck-iteration3-ast.md`
Method: TDD, task-by-task (executing-plans skill). 6 commits, 85 codecheck unit tests passing.
Live three-way result: `docs/reports/07-codecheck-iteration3-ast-result.md`.

## Goal

Add the third code-domain SelfCheck variant — **structural consistency via AST** — and
produce the three-way Exec / Prompt / AST comparison on the same seed-1 MBPP+ sample, so
all three are judged on identical generated code. The driving question: **does syntactic
structure add signal beyond behavior (Exec) and the judge (Prompt), or does it share Exec's
confident-consistent blind spot** (consistently-wrong code whose samples are also
structurally similar to the wrong main)?

This iteration also folded in and closed the **iteration-2.5 validation gate**: the same
`--method all` run re-establishes trustworthy post-hashseed-fix Exec/Prompt numbers.

## Features introduced

- **SelfCheck-AST scorer** (`codecheck/ast_score.py`) — pure local, **no model / no API
  cost** at scoring time (stdlib `ast` + `collections.Counter` only):
  - `ast_fingerprint(code)` — multiset (Counter) of AST node-type names (`Name`, `BinOp`,
    `Constant`, …). Counts only node *type*, never identifier text or literal value, so it
    is **invariant to variable renaming and literal changes**. Returns `None` on a
    `SyntaxError`.
  - `ast_dissimilarity(a, b)` — `1 − multiset Jaccard` of two fingerprints: `0.0` =
    identical structure, up to `1.0` = no shared node types. Two empty fingerprints count
    as identical.
  - `ASTScorer` — `evaluate(main, samples)` returns `(mean_dissimilarity, per_sample)` on
    the same `[0,1]` scale and direction as Exec/Prompt (higher = more likely incorrect).
    Unparseable sample → `1.0` (max divergence); unparseable main → all samples `1.0`.
    Accumulates a `parse_failures` count, mirroring `PromptJudge`.
- **Pipeline wiring** (`codecheck/pipeline.py`) — `score_problem` / `run_dataset` take an
  `ast_scorer` parameter and an `"ast"` scoring branch; raises `ValueError` if `"ast"` is
  requested without a scorer. AST scores the **main implementation's** structural
  consistency against its N samples (per-problem unit, label = main's execution
  correctness) — same unit as Exec and Prompt.
- **`run --method {exec,prompt,ast,both,all}`** — `ast` = AST only; `all` = exec + prompt +
  ast. After a run the CLI prints `AST parse failures: M` alongside the judge count.
- **No changes to `models.py` or `evaluate.py`.** The `scores` dict and the method-generic
  readout (`_methods_present`, `score_histogram`, `auc_pr_detect_incorrect`) already
  accommodate a new method name, so `[ast]` appears in the `evaluate` output automatically.

## Deliverables

- `codecheck/ast_score.py` (new) — fingerprint, dissimilarity, `ASTScorer`.
- `codecheck/pipeline.py` — `ast_scorer` param + `"ast"` branch in `score_problem`/`run_dataset`.
- `run_codecheck.py` — `--method ast/all`, scorer wiring, AST parse-failure print.
- `codecheck/README.md` — SelfCheck-AST method explanation + `--method ast/all` usage.
- Tests: `test_codecheck_ast_score.py` (new, 10) + pipeline (2) + CLI (3) additions.

## Testable conditions (all covered by unit tests, 85 codecheck tests passing)

- Fingerprint invariant to variable renaming and to literal-value changes.
- Fingerprint returns `None` on a syntax error; identical structure → `0.0` dissimilarity,
  different structure → `(0, 1]`.
- `ASTScorer` means per-sample dissimilarities; unparseable sample / main → `1.0` and counted.
- Empty sample list → `0.0`.
- Pipeline `"ast"` branch fills `scores["ast"]`, skips sample execution in an ast-only run,
  still labels correctness; missing scorer raises `ValueError`.
- Parser accepts `ast` and `all`; rejects unknown choices.

## User test flow (live — needs `OPENROUTER_API_KEY`)

Generation hits OpenRouter (the main + N samples); AST scoring itself adds no API calls.

1. **Run all three methods on the standing-protocol sample** (random full-set, capable
   model, seed for reproducible problem selection):
   ```bash
   python run_codecheck.py run --method all --limit 30 --n 5 --seed 1 --timeout 5 \
     --output output/iter3-all.json
   ```
   Each per-problem line now shows `exec=… prompt=… ast=…`. At the end, watch
   `Judge parse failures: N` and `AST parse failures: M` (both should be ~0).

2. **Read the three-way comparison:**
   ```bash
   python run_codecheck.py evaluate --results output/iter3-all.json
   ```
   You get an `[exec]`, `[prompt]`, and `[ast]` block, each with trapezoidal AUC-PR, the
   shared prevalence baseline, and a per-class score histogram.

3. **What to look for (the iteration's core questions):**
   - Read each AUC-PR **against the baseline**, not in isolation. (This run: exec 0.696 >
     ast 0.591 > prompt 0.534 vs 0.300 floor.)
   - **Blind-spot check:** in the histograms, find incorrect mains Exec puts in `[0.0,0.1)`
     *and* Prompt also scores ≈0 (confident-consistent misses). Look up their `ast` score —
     does structure add lift, or also miss them? (This run: `Mbpp/99`, `Mbpp/310` got
     ast 0.208 / 0.167 — small lift, partially refuting the shared-blind-spot hypothesis.)
   - **AST false positives:** scan correct mains with high `ast` (the `[0.3,1.0)` bins,
     correct column) — structurally-diverse-but-correct code. This is AST's precision cost.

4. **Sanity / robustness (optional):**
   ```bash
   python run_codecheck.py run --method ast --limit 5 --n 3      # ast-only; still generates, no judge cost
   python run_codecheck.py evaluate --results /tmp/nope.json     # clean error, exit 1
   ```

## Result summary (from report 07)

- All three beat the 0.300 baseline: **exec 0.696 > ast 0.591 > prompt 0.534** (n=30, 9
  incorrect, 0 parse failures).
- AST is a **complementary third axis** — beats Prompt here, clean low-end specificity
  (every incorrect main scored `ast ≥ 0.156`; no incorrect in `[0.0,0.1)`).
- Bag-of-node-types Jaccard **discriminates** — the MVP metric is informative, not
  degenerate; tree-edit-distance not justified yet.
- AST's weakness is **false positives** (4 correct mains with `ast ≥ 0.3`) — same family as
  Exec's adversarial-input false positives → joint iteration-4 hardening target.

## Caveat — stochastic generation across runs

`--seed` fixes *problem selection* only; code generation is stochastic (temperature
sampling, no generation seed). So probe scores do not reproduce verbatim across runs —
report 07 uses *that run's own* confident-consistent cases, not the plan's stale ones.

## Feedback to collect (feeds iteration 4)

- Does AST add signal beyond Exec/Prompt, or track Exec? → **adds a third axis; beats
  Prompt here, below Exec.**
- Does AST confirm the shared confident-consistent blind spot? → **partially refuted; small
  lift on the exec=0&prompt=0 cases.**
- Which AST metric behaves best? → **bag-of-node-types Jaccard kept; richer metric deferred.**
- AST parse-failure rate at this N → **0.**

## Open / deferred

- **Iteration 4:** joint Exec + AST false-positive hardening (reference-vs-sample input
  normalization); scale to full set + larger N for paper-comparable numbers.
- Structure-sensitive AST metric (tree edit distance, subtree n-grams) — deferred unless
  false-positive hardening alone leaves AST behind Exec.

Part of the iteration roadmap (`docs/plans/01-codecheck-roadmap.md`).
