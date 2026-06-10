# SelfCheck-AST (Iteration 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the third code-domain SelfCheck variant — structural consistency via AST — and produce the three-way Exec / Prompt / AST comparison on the same MBPP+ sample.

**Architecture:** A new `codecheck/ast_score.py` computes a rename- and literal-invariant structural fingerprint of each implementation (multiset of Python AST node types) and scores each sample's dissimilarity from the main, aggregated to a `[0,1]` inconsistency score on the *same scale and direction* as Exec and Prompt (higher = more likely incorrect). It wires into the existing method-generic pipeline exactly like the Prompt judge: a scorer object with a `parse_failures` accumulator, passed into `score_problem`. Because `CodeResult.scores`, `evaluate.py`, and the CLI readout are already method-agnostic, **no changes to `models.py` or `evaluate.py` are needed** — `[ast]` appears in the readout automatically once results carry an `ast` score.

**Tech Stack:** Python stdlib `ast` + `collections.Counter` (no new dependency), pytest, existing `codecheck/` package, OpenRouter via `run_codecheck.py`.

---

## Context — what feeds this iteration

This plan folds in the iteration-2.5 validation-run feedback
(`docs/plans/01-codecheck-roadmap.md`, Revision 2) and the resolved unit
granularity (`docs/source/04-codecheck-methods.md`):

- **Unit = per-problem / per-main-implementation.** The scored unit is the single
  `T=0` main implementation; the `N` samples are evidence only. AST follows this:
  it scores the *main* impl's structural consistency against its samples, label =
  the main's execution correctness. Same as Exec and Prompt.
- **AST's hypothesis to test:** structure shares Exec's *confident-consistent blind
  spot* — when a model is consistently wrong, the samples are also structurally
  similar to the (wrong) main, so AST scores ≈0 and misses it, just like Exec.
  The 2.5 run gives concrete probe cases on the seed-1 sample:
  - **`Mbpp/237`** — incorrect, Exec=0.000 **and** Prompt=0.000: a true
    confident-consistent miss. **Does AST also score it ≈0?** (Expected: yes — the
    hypothesis.)
  - **`Mbpp/785`** — incorrect, Exec=0.026 (misses) but Prompt=1.000 (catches):
    the judge complements Exec here. **Does AST add anything, or track Exec?**
  - **`Mbpp/577`** — incorrect, Exec=0.746, Prompt=1.000: both catch. AST sanity:
    should also score high if structure diverges on genuinely-divergent samples.
- **Standing protocol (carried from iter 1):** every readout prints the prevalence
  baseline + per-class score histogram next to the trapezoidal AUC-PR. The CLI
  already does this for all methods; AST inherits it for free.

The iteration's deliverable is the **three-way readout on the same seed-1 sample**
the 2.5 run used, so all three methods are compared on identical data.

## File Structure

- **Create** `codecheck/ast_score.py` — fingerprint + dissimilarity functions and
  the `ASTScorer` class (mirrors `prompt_score.py`'s function-plus-class shape).
- **Create** `tests/test_codecheck_ast_score.py` — unit tests for the new module.
- **Modify** `codecheck/pipeline.py` — add an `ast_scorer` parameter to
  `score_problem` / `run_dataset` and an `"ast"` scoring branch.
- **Modify** `tests/test_codecheck_pipeline.py` — add an AST-path test.
- **Modify** `run_codecheck.py` — add `ast` / `all` to `--method`, instantiate the
  scorer, pass it through, and print AST parse failures.
- **Modify** `codecheck/README.md` — document `--method ast` / `all`.

No change to `codecheck/models.py` or `codecheck/evaluate.py`: the `scores` dict and
the method-generic readout already accommodate a new method name.

---

### Task 1: AST fingerprint + dissimilarity (pure functions)

**Files:**
- Create: `codecheck/ast_score.py`
- Test: `tests/test_codecheck_ast_score.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_codecheck_ast_score.py
from codecheck.ast_score import ast_fingerprint, ast_dissimilarity


def test_fingerprint_is_invariant_to_variable_renaming():
    a = ast_fingerprint("def f(x):\n    return x + 1\n")
    b = ast_fingerprint("def f(y):\n    return y + 1\n")
    assert a == b


def test_fingerprint_is_invariant_to_literal_values():
    a = ast_fingerprint("def f(x):\n    return x + 1\n")
    b = ast_fingerprint("def f(x):\n    return x + 99\n")
    assert a == b


def test_fingerprint_returns_none_on_syntax_error():
    assert ast_fingerprint("@@@ not python @@@") is None


def test_identical_structure_has_zero_dissimilarity():
    a = ast_fingerprint("def f(x):\n    return x + 1\n")
    assert ast_dissimilarity(a, a) == 0.0


def test_different_structure_has_positive_dissimilarity():
    a = ast_fingerprint("def f(x):\n    return x + 1\n")
    b = ast_fingerprint("def f(x):\n    for i in range(x):\n        print(i)\n")
    d = ast_dissimilarity(a, b)
    assert 0.0 < d <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_codecheck_ast_score.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'codecheck.ast_score'`

- [ ] **Step 3: Write the minimal implementation**

```python
# codecheck/ast_score.py
from __future__ import annotations
import ast
from collections import Counter


def ast_fingerprint(code: str) -> Counter | None:
    """Multiset of AST node-type names for `code`, or None if it does not parse.

    Only the node *type* is counted (e.g. `Name`, `BinOp`, `Constant`), never the
    identifier text or literal value — so renaming variables or changing constants
    leaves the fingerprint unchanged. This captures structure, normalized for
    identifier renaming and literal differences (the methods-doc requirement).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    return Counter(type(node).__name__ for node in ast.walk(tree))


def ast_dissimilarity(main_fp: Counter, sample_fp: Counter) -> float:
    """1 - multiset Jaccard of two fingerprints. 0.0 = identical structure,
    1.0 = no shared node types. Two empty fingerprints count as identical."""
    intersection = sum((main_fp & sample_fp).values())
    union = sum((main_fp | sample_fp).values())
    if union == 0:
        return 0.0
    return 1.0 - intersection / union
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_codecheck_ast_score.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add codecheck/ast_score.py tests/test_codecheck_ast_score.py
git commit -m "feat(ast): rename-invariant AST fingerprint + Jaccard dissimilarity"
```

---

### Task 2: ASTScorer class (parse-failure accumulator)

**Files:**
- Modify: `codecheck/ast_score.py`
- Test: `tests/test_codecheck_ast_score.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_codecheck_ast_score.py
from codecheck.ast_score import ASTScorer


def test_scorer_mean_dissimilarity_over_samples():
    main = "def f(x):\n    return x + 1\n"
    scorer = ASTScorer()
    # identical sample -> 0.0; structurally different sample -> > 0.0; mean is between.
    score, per_sample = scorer.evaluate(main, [main, "def f(x):\n    while x:\n        x -= 1\n    return x\n"])
    assert per_sample[0] == 0.0
    assert per_sample[1] > 0.0
    assert abs(score - sum(per_sample) / 2) < 1e-9
    assert scorer.parse_failures == 0


def test_scorer_counts_unparseable_sample_as_max_divergence():
    main = "def f(x):\n    return x + 1\n"
    scorer = ASTScorer()
    score, per_sample = scorer.evaluate(main, ["@@@ not python @@@"])
    assert per_sample == [1.0]
    assert score == 1.0
    assert scorer.parse_failures == 1


def test_scorer_unparseable_main_scores_max_and_counts_once():
    scorer = ASTScorer()
    score, per_sample = scorer.evaluate("@@@ bad main @@@", ["def f(): return 1", "def g(): return 2"])
    assert score == 1.0
    assert per_sample == [1.0, 1.0]
    assert scorer.parse_failures == 1


def test_scorer_empty_samples_scores_zero():
    scorer = ASTScorer()
    assert scorer.score("def f(): return 1", []) == 0.0


def test_scorer_score_wraps_evaluate():
    main = "def f(x):\n    return x + 1\n"
    scorer = ASTScorer()
    assert scorer.score(main, [main]) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_codecheck_ast_score.py -v`
Expected: FAIL — `ImportError: cannot import name 'ASTScorer'`

- [ ] **Step 3: Write the minimal implementation**

Append to `codecheck/ast_score.py`:

```python
class ASTScorer:
    """Structural-consistency scorer (SelfCheck-AST). evaluate() returns the mean
    structural dissimilarity of the samples vs the main implementation, on the same
    [0,1] scale and direction as Exec/Prompt (higher = more likely incorrect).
    `parse_failures` accumulates implementations that do not parse, mirroring
    PromptJudge so the CLI can report it.
    """

    def __init__(self) -> None:
        self.parse_failures = 0

    def evaluate(self, main_code: str, sample_codes: list[str]) -> tuple[float, list[float]]:
        """(mean_dissimilarity, per_sample_dissimilarities)."""
        if not sample_codes:
            return 0.0, []
        main_fp = ast_fingerprint(main_code)
        if main_fp is None:
            # The T=0 main failed to parse: structure is unverifiable, so treat every
            # sample as maximally divergent and count one parse failure for the main.
            self.parse_failures += 1
            return 1.0, [1.0] * len(sample_codes)
        per_sample: list[float] = []
        for code in sample_codes:
            sample_fp = ast_fingerprint(code)
            if sample_fp is None:
                self.parse_failures += 1
                per_sample.append(1.0)
            else:
                per_sample.append(ast_dissimilarity(main_fp, sample_fp))
        return sum(per_sample) / len(per_sample), per_sample

    def score(self, main_code: str, sample_codes: list[str]) -> float:
        return self.evaluate(main_code, sample_codes)[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_codecheck_ast_score.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add codecheck/ast_score.py tests/test_codecheck_ast_score.py
git commit -m "feat(ast): ASTScorer with parse-failure accumulation"
```

---

### Task 3: Wire AST into the pipeline

**Files:**
- Modify: `codecheck/pipeline.py:22-47` (`score_problem`), `codecheck/pipeline.py:50-64` (`run_dataset`)
- Test: `tests/test_codecheck_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_codecheck_pipeline.py
from codecheck.ast_score import ASTScorer


def test_score_problem_fills_ast():
    gen = StubGen("def f(x):\n    return x + 1\n",
                  ["def f(x):\n    return x + 1\n", "def f(x):\n    return x + 1\n"])
    res = score_problem(PROBLEM, gen, run_batch_in_subprocess, n_samples=2, timeout=5.0,
                        methods={"ast"}, ast_scorer=ASTScorer())
    assert "ast" in res.scores
    assert "exec" not in res.scores          # ast-only run skips sample execution
    assert res.scores["ast"] == 0.0          # samples structurally identical to main
    assert res.is_correct is True            # labeling still runs (main vs canonical)
    assert res.n_inputs == 3


def test_score_problem_ast_requires_scorer():
    import pytest
    gen = StubGen("def f(x):\n    return x + 1\n", ["def f(x):\n    return x + 1\n"])
    with pytest.raises(ValueError):
        score_problem(PROBLEM, gen, run_batch_in_subprocess, n_samples=1, timeout=5.0,
                      methods={"ast"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_codecheck_pipeline.py -v`
Expected: FAIL — `TypeError: score_problem() got an unexpected keyword argument 'ast_scorer'`

- [ ] **Step 3: Modify `score_problem` and `run_dataset`**

In `codecheck/pipeline.py`, change the `score_problem` signature and add the AST branch:

```python
def score_problem(problem, generator, harness, n_samples: int, timeout: float = 5.0,
                  methods: set[str] | None = None, judge=None, ast_scorer=None) -> CodeResult:
    methods = methods or {"exec"}
    main_code, sample_codes = generator.generate(problem, n_samples)
    main_outputs = _run_vector(main_code, problem, harness, timeout)
    expected = expected_outputs(problem, harness, timeout)

    scores: dict[str, float] = {}
    prompt_responses: list[str] | None = None
    if "exec" in methods:
        sample_outputs = [_run_vector(code, problem, harness, timeout) for code in sample_codes]
        scores["exec"] = exec_inconsistency(main_outputs, sample_outputs)
    if "prompt" in methods:
        if judge is None:
            raise ValueError("method 'prompt' requires a judge")
        scores["prompt"], prompt_responses = judge.evaluate(main_code, sample_codes)
    if "ast" in methods:
        if ast_scorer is None:
            raise ValueError("method 'ast' requires an ast_scorer")
        scores["ast"], _ = ast_scorer.evaluate(main_code, sample_codes)

    return CodeResult(
        task_id=problem.task_id,
        scores=scores,
        is_correct=is_correct(main_outputs, expected),
        main_code=main_code,
        sample_codes=sample_codes,
        n_inputs=len(problem.inputs),
        prompt_responses=prompt_responses,
    )
```

Then thread `ast_scorer` through `run_dataset`:

```python
def run_dataset(problems, generator, harness, n_samples: int, timeout: float = 5.0,
                methods: set[str] | None = None, judge=None, ast_scorer=None) -> list[CodeResult]:
    problems = list(problems)
    total = len(problems)
    results: list[CodeResult] = []
    for i, problem in enumerate(tqdm(problems, desc="codecheck"), start=1):
        started = time.monotonic()
        result = score_problem(problem, generator, harness, n_samples, timeout, methods, judge, ast_scorer)
        elapsed = time.monotonic() - started
        scores = "  ".join(f"{name}={value:.3f}" for name, value in result.scores.items())
        tqdm.write(f"[{i}/{total}] {result.task_id}  correct={result.is_correct}  "
                   f"{scores}  n_inputs={result.n_inputs}  ({elapsed:.1f}s)")
        results.append(result)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_codecheck_pipeline.py -v`
Expected: PASS (all pipeline tests, including the two new ones)

- [ ] **Step 5: Commit**

```bash
git add codecheck/pipeline.py tests/test_codecheck_pipeline.py
git commit -m "feat(ast): wire ASTScorer into score_problem/run_dataset"
```

---

### Task 4: CLI — `--method ast` and `--method all`

**Files:**
- Modify: `run_codecheck.py:34-67` (`_cmd_run`), `run_codecheck.py:122-123` (`--method` choices)
- Test: `tests/test_codecheck_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_codecheck_cli.py
def test_method_ast_is_accepted_by_parser():
    from run_codecheck import build_parser
    args = build_parser().parse_args(["run", "--method", "ast"])
    assert args.method == "ast"


def test_method_all_is_accepted_by_parser():
    from run_codecheck import build_parser
    args = build_parser().parse_args(["run", "--method", "all"])
    assert args.method == "all"
```

> Note: read the existing `tests/test_codecheck_cli.py` first to match its import
> style and any shared fixtures; append these two tests in the same style.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_codecheck_cli.py -v`
Expected: FAIL — argparse raises `SystemExit` on the unknown `ast`/`all` choice.

- [ ] **Step 3: Update the CLI**

In `run_codecheck.py`, add the import near the other scorer imports in `_cmd_run`:

```python
    from codecheck.ast_score import ASTScorer
```

Replace the method-selection + scorer-construction block in `_cmd_run`:

```python
    if args.method == "all":
        methods = {"exec", "prompt", "ast"}
    elif args.method == "both":
        methods = {"exec", "prompt"}
    else:
        methods = {args.method}
    judge = PromptJudge(client, model=model, think=args.think) if "prompt" in methods else None
    ast_scorer = ASTScorer() if "ast" in methods else None
```

Pass `ast_scorer` into `run_dataset`:

```python
        results = run_dataset(problems, generator, run_batch_in_subprocess,
                              n_samples=args.n, timeout=args.timeout,
                              methods=methods, judge=judge, ast_scorer=ast_scorer)
```

After the `Judge parse failures` print, add the AST report:

```python
    if ast_scorer is not None:
        print(f"AST parse failures: {ast_scorer.parse_failures}")
```

Widen the `--method` choices:

```python
    run_p.add_argument("--method", choices=["exec", "prompt", "ast", "both", "all"], default="exec",
                       help="which consistency scorer(s) to run")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_codecheck_cli.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `pytest tests/ -q`
Expected: PASS — prior suite (58) + new AST/pipeline/CLI tests, all green.

- [ ] **Step 6: Commit**

```bash
git add run_codecheck.py tests/test_codecheck_cli.py
git commit -m "feat(ast): add --method ast/all and AST parse-failure reporting"
```

---

### Task 5: Document the AST variant

**Files:**
- Modify: `codecheck/README.md`

- [ ] **Step 1: Read the README to find the `--method` section**

Run: `grep -n "method" codecheck/README.md`

- [ ] **Step 2: Add AST usage**

Add `ast` and `all` to the `--method` documentation and a one-line example, in the
same style as the existing `exec`/`prompt`/`both` entries. Show the three-way run:

```bash
python run_codecheck.py run --method all --limit 30 --n 5 --seed 1 --timeout 5 \
  --output output/iter3-all.json
python run_codecheck.py evaluate --results output/iter3-all.json
```

Note that AST adds no API cost (pure local parsing) and that AST parse failures are
reported alongside judge parse failures.

- [ ] **Step 3: Commit**

```bash
git add codecheck/README.md
git commit -m "docs(ast): document --method ast/all in codecheck README"
```

---

### Task 6: Live three-way validation run (the iteration's readout)

> This is the user-facing deliverable: the three-way comparison on the SAME seed-1
> sample the iter-2.5 run used, so Exec / Prompt / AST are judged on identical data.
> Needs `OPENROUTER_API_KEY` (already in `.env`).

- [ ] **Step 1: Run all three methods on the seed-1 sample**

```bash
python run_codecheck.py run --method all --limit 30 --n 5 --seed 1 --timeout 5 \
  --output output/iter3-all.json
```
Expected: completes; prints `Judge parse failures: N` and `AST parse failures: M`
(both should be low). The per-problem lines now show `exec=… prompt=… ast=…`.

- [ ] **Step 2: Read the three-way comparison**

```bash
python run_codecheck.py evaluate --results output/iter3-all.json
```
Expected: a `[exec]`, `[prompt]`, and `[ast]` block, each with trapezoidal AUC-PR,
the shared prevalence baseline, and a per-class histogram.

- [ ] **Step 3: Answer the iteration's questions (write a short report)**

Create `docs/reports/07-codecheck-iteration3-ast-result.md` capturing:
- The three AUC-PR numbers vs the prevalence baseline (does AST add signal beyond
  Exec/Prompt, or track Exec?).
- **The blind-spot check:** look up `Mbpp/237` (Exec=0 *and* Prompt=0 on the 2.5
  run) in the new results — what is its `ast` score? If ≈0, AST confirms the
  hypothesis that structure shares Exec's confident-consistent blind spot.
- `Mbpp/785` (Exec misses, Prompt catches): does AST track Exec (≈0) or the judge?
- AST parse-failure rate (feeds whether the fingerprint metric needs hardening).
- A recommendation on the AST metric: does bag-of-node-types Jaccard discriminate,
  or is a structure-sensitive metric (tree edit distance, subtree n-grams) needed?
  This is the iteration-4 / methods-doc feedback.

- [ ] **Step 4: Commit the report**

```bash
git add docs/reports/07-codecheck-iteration3-ast-result.md output/iter3-all.json
git commit -m "report(ast): three-way Exec/Prompt/AST comparison on seed-1 MBPP+"
```

---

## Feedback to collect (feeds iteration 4 + the report)

- Does AST add signal beyond Exec/Prompt, or is it redundant with Exec?
- Does AST confirm the shared confident-consistent blind spot (`Mbpp/237` ≈0)?
- Which AST metric behaves best — is bag-of-node-types Jaccard enough, or is a
  structure-sensitive metric (tree edit distance) needed? (Deferred unless the MVP
  metric is uninformative; decided here from the readout.)
- AST parse-failure rate at this N.

## Risks / open decisions

- **Metric coarseness.** Bag-of-node-types Jaccard ignores nesting and ordering: two
  structurally different programs with the same node-type counts score identical.
  This is the intended MVP — cheap, debuggable, dependency-free. If Task 6's readout
  shows it is uninformative (e.g. AST AUC-PR ≈ baseline with no discrimination in the
  histogram), the follow-up is a structure-sensitive metric (tree edit distance via a
  library, or AST-path / subtree n-grams), planned when iteration 4 is detailed — not
  added speculatively now.

## Self-review notes

- Spec coverage: methods-doc AST open points are all addressed — metric choice
  (bag Jaccard, with TED named as the deferred alternative), rename/literal
  normalization (node-type-only fingerprint, Task 1), parse-failure handling
  (Task 2). Three-way comparison is Task 6.
- No `models.py` / `evaluate.py` edits required — verified the `scores` dict and
  `_methods_present`/`score_histogram`/`auc_pr_detect_incorrect` are method-generic.
- Type consistency: `ASTScorer.evaluate` returns `(float, list[float])`, mirroring
  `PromptJudge.evaluate`'s `(float, list[str])`; `score_problem` discards the second
  element (`scores["ast"], _ = ...`) just as it keeps `prompt_responses` for prompt.
