# SelfCheck-Prompt Variant (Iteration 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the SelfCheck-Prompt (LLM-as-judge) consistency variant alongside SelfCheck-Exec, scoring the same generated implementations so Exec vs Prompt AUC-PR can be compared head-to-head on identical MBPP+ data — specifically to test whether the judge catches the confident-consistent hallucinations Exec scores ≈0 on.

**Architecture:** Generalize `CodeResult` to hold a `scores: dict[str, float]` (method name → score), so one `run` generates implementations once and scores them with every selected method. The judge compares each sample against the main implementation, returns Yes/No/N-A per sample, and aggregates to a per-implementation inconsistency score on the same `[0, 1]` scale as Exec (higher = more likely hallucinated). `evaluate` is generalized to select a method and to print, for every method present, the trapezoidal AUC-PR plus the prevalence baseline and a per-class score histogram.

**Tech Stack:** Python 3.13, OpenAI SDK against OpenRouter, scikit-learn (existing `precision_recall_curve`/`auc`), pytest, `ThreadPoolExecutor` for concurrent judge calls.

---

## Background for the implementing engineer

You have not seen this codebase. Read these first:

- `codecheck/models.py` — `CodeProblem` and `CodeResult` dataclasses with `to_dict`/`from_dict`.
- `codecheck/exec_score.py` — `exec_inconsistency(main_outputs, sample_outputs)`: `1 - mean per-sample output agreement`, range `[0,1]`. Higher = more divergent = more likely incorrect. The Prompt score uses the **same direction and range**.
- `codecheck/evaluate.py` — `auc_pr_detect_incorrect(results)`: labels `1` for incorrect mains, ranks by score, trapezoidal AUC-PR. `<2` classes → `nan`.
- `codecheck/generation.py` — `CodeGenerator`: concurrent `_complete` calls; `extra_body={"reasoning": {"enabled": self.think}}` disables hidden chain-of-thought (≈20× faster). The judge reuses this exact pattern.
- `codecheck/pipeline.py` — `score_problem` / `run_dataset` orchestrate generate → execute → score → label.
- `run_codecheck.py` — CLI with `run` and `evaluate` subcommands.
- Reference for the original NL judge: `replication/score/prompt.py` (template + Yes/No mapping).

**Conventions to follow:**
- `from __future__ import annotations` at the top of every module.
- Dataclasses with explicit `to_dict`/`from_dict`.
- Tests live in `tests/test_codecheck_*.py`, use plain `pytest` functions and a `FakeClient` built from `types.SimpleNamespace` (see `tests/test_codecheck_generation.py`).
- Keep public entry points high-level; push API/parse mechanics into small private helpers.
- Run the whole suite with `pytest tests/ -q`.
- Commit after every task. No `Co-Authored-By` / no Claude footer (project rule).

**Score-direction invariant (must hold for every method):** higher score = more likely incorrect. Exec: divergence. Prompt: judged inconsistency. Mapping `Yes(consistent)→0.0`, `No(inconsistent)→1.0`, `N-A→0.5`, aggregated by mean over the N samples.

---

## File Structure

- **Modify** `codecheck/models.py` — `CodeResult.exec_score` field → `scores: dict[str, float]`; add `exec_score` read property; back-compat `from_dict`.
- **Modify** `codecheck/evaluate.py` — `auc_pr_detect_incorrect(results, method="exec")`; add `prevalence_baseline` and `score_histogram`.
- **Create** `codecheck/prompt_score.py` — `build_judge_prompt`, `parse_judgment`, `PromptJudge`.
- **Modify** `codecheck/pipeline.py` — `score_problem` / `run_dataset` accept `methods` set + optional `judge`; write the `scores` dict.
- **Modify** `run_codecheck.py` — `run --method {exec,prompt,both}` builds the judge and scores all selected methods; `evaluate` prints a per-method comparison (AUC-PR + baseline + histogram).
- **Modify tests:** `tests/test_codecheck_models.py`, `tests/test_codecheck_evaluate.py`, `tests/test_codecheck_pipeline.py`, `tests/test_codecheck_cli.py`.
- **Create test:** `tests/test_codecheck_prompt_score.py`.
- **Modify** `codecheck/README.md` — document `--method` and the standing run/report protocol.

---

## Task 1: Generalize `CodeResult` to a per-method `scores` dict

**Files:**
- Modify: `codecheck/models.py`
- Test: `tests/test_codecheck_models.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_codecheck_models.py`:

```python
from codecheck.models import CodeResult


def test_coderesult_scores_dict_roundtrip():
    r = CodeResult("t", {"exec": 0.3, "prompt": 0.7}, False, "main", ["s1"])
    d = r.to_dict()
    assert d["scores"] == {"exec": 0.3, "prompt": 0.7}
    back = CodeResult.from_dict(d)
    assert back.scores == {"exec": 0.3, "prompt": 0.7}
    assert back.is_correct is False


def test_coderesult_exec_score_property():
    r = CodeResult("t", {"exec": 0.42}, True, "m", [])
    assert r.exec_score == 0.42


def test_coderesult_from_legacy_exec_score_key():
    # iteration-1 artifacts stored a bare "exec_score" key, no "scores"
    legacy = {"task_id": "t", "exec_score": 0.358, "is_correct": False,
              "main_code": "m", "sample_codes": ["s"]}
    r = CodeResult.from_dict(legacy)
    assert r.scores == {"exec": 0.358}
    assert r.exec_score == 0.358
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_codecheck_models.py -q`
Expected: FAIL (`CodeResult.__init__` still takes a float `exec_score`; no `scores`).

- [ ] **Step 3: Rewrite `CodeResult`**

Replace the `CodeResult` dataclass in `codecheck/models.py` with:

```python
@dataclass
class CodeResult:
    task_id: str
    scores: dict[str, float]      # method name -> inconsistency score in [0, 1]
    is_correct: bool
    main_code: str
    sample_codes: list[str]

    @property
    def exec_score(self) -> float:
        return self.scores.get("exec", float("nan"))

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "scores": self.scores, "is_correct": self.is_correct,
            "main_code": self.main_code, "sample_codes": self.sample_codes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CodeResult":
        if "scores" in d:
            scores = dict(d["scores"])
        elif "exec_score" in d:               # iteration-1 artifact back-compat
            scores = {"exec": d["exec_score"]}
        else:
            scores = {}
        return cls(
            task_id=d["task_id"], scores=scores, is_correct=d["is_correct"],
            main_code=d["main_code"], sample_codes=d["sample_codes"],
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_codecheck_models.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add codecheck/models.py tests/test_codecheck_models.py
git commit -m "refactor(codecheck): CodeResult stores per-method scores dict"
```

---

## Task 2: Generalize `evaluate` + add baseline and histogram

**Files:**
- Modify: `codecheck/evaluate.py`
- Test: `tests/test_codecheck_evaluate.py`

The existing tests build `CodeResult` positionally with a float second arg — they must move to the `scores` dict. AUC-PR gains a `method` selector. Add `prevalence_baseline` (the PR no-skill floor = positive prevalence) and `score_histogram` (per-class counts per bin) — the rigor items iteration 1 flagged.

- [ ] **Step 1: Rewrite the failing tests**

Replace the body of `tests/test_codecheck_evaluate.py` with:

```python
import math
from codecheck.models import CodeResult
from codecheck.evaluate import (
    auc_pr_detect_incorrect, prevalence_baseline, score_histogram,
)


def _r(score, correct, method="exec"):
    return CodeResult("t", {method: score}, correct, "m", ["s"])


def test_perfect_separation_scores_one():
    results = [_r(0.9, False), _r(0.8, False), _r(0.1, True), _r(0.0, True)]
    assert auc_pr_detect_incorrect(results) == 1.0


def test_single_class_is_nan():
    results = [_r(0.5, True), _r(0.4, True)]
    assert math.isnan(auc_pr_detect_incorrect(results))


def test_auc_selects_named_method():
    # exec ranks perfectly; prompt ranks inverted on the same results
    results = [
        CodeResult("a", {"exec": 0.9, "prompt": 0.0}, False, "m", []),
        CodeResult("b", {"exec": 0.1, "prompt": 1.0}, True, "m", []),
    ]
    assert auc_pr_detect_incorrect(results, method="exec") == 1.0
    assert auc_pr_detect_incorrect(results, method="prompt") < 1.0


def test_prevalence_baseline_is_positive_fraction():
    results = [_r(0.5, False), _r(0.5, False), _r(0.5, True), _r(0.5, True)]
    assert prevalence_baseline(results) == 0.5


def test_score_histogram_splits_by_class():
    results = [_r(0.05, True), _r(0.05, True), _r(0.95, False)]
    hist = score_histogram(results, method="exec", bins=10)
    assert len(hist) == 10
    first = hist[0]      # bucket [0.0, 0.1)
    last = hist[-1]      # bucket [0.9, 1.0]
    assert first["correct"] == 2 and first["incorrect"] == 0
    assert last["incorrect"] == 1 and last["correct"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_codecheck_evaluate.py -q`
Expected: FAIL (`method` kwarg, `prevalence_baseline`, `score_histogram` undefined).

- [ ] **Step 3: Rewrite `codecheck/evaluate.py`**

```python
from __future__ import annotations
from sklearn.metrics import auc, precision_recall_curve

from codecheck.models import CodeResult


def auc_pr_detect_incorrect(results: list[CodeResult], method: str = "exec") -> float:
    y_true = [0 if r.is_correct else 1 for r in results]
    scores = [r.scores[method] for r in results]
    if len(set(y_true)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return float(auc(recall, precision))


def prevalence_baseline(results: list[CodeResult]) -> float:
    """PR no-skill floor = fraction of incorrect (positive) mains."""
    if not results:
        return float("nan")
    return sum(1 for r in results if not r.is_correct) / len(results)


def score_histogram(results: list[CodeResult], method: str = "exec", bins: int = 10) -> list[dict]:
    """Per-bin counts split by true class. Bin i covers [i/bins, (i+1)/bins);
    the final bin is closed on the right so score == 1.0 lands in it."""
    hist = [{"lo": i / bins, "hi": (i + 1) / bins, "correct": 0, "incorrect": 0}
            for i in range(bins)]
    for r in results:
        s = r.scores[method]
        idx = min(int(s * bins), bins - 1)
        key = "correct" if r.is_correct else "incorrect"
        hist[idx][key] += 1
    return hist
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_codecheck_evaluate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add codecheck/evaluate.py tests/test_codecheck_evaluate.py
git commit -m "feat(codecheck): evaluate selects method, adds baseline + histogram"
```

---

## Task 3: Judge prompt + answer parsing

**Files:**
- Create: `codecheck/prompt_score.py`
- Test: `tests/test_codecheck_prompt_score.py`

`parse_judgment` maps the judge's free text to a score and reports whether it matched a real Yes/No/N-A token (so parse failures can be counted). `build_judge_prompt` assembles the template from `04-codecheck-methods.md`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_codecheck_prompt_score.py`:

```python
from codecheck.prompt_score import build_judge_prompt, parse_judgment


def test_build_judge_prompt_includes_both_codes():
    p = build_judge_prompt("def f(): return 1", "def f(): return 2")
    assert "def f(): return 1" in p
    assert "def f(): return 2" in p
    assert "Yes" in p and "No" in p


def test_parse_yes_means_consistent_zero():
    score, matched = parse_judgment("Yes, the behavior is identical.")
    assert score == 0.0 and matched is True


def test_parse_no_means_inconsistent_one():
    score, matched = parse_judgment("No - it differs on negative inputs.")
    assert score == 1.0 and matched is True


def test_parse_na_is_half():
    score, matched = parse_judgment("N/A because the construct is unrelated.")
    assert score == 0.5 and matched is True


def test_parse_unmatched_is_half_and_unmatched_flag():
    score, matched = parse_judgment("I cannot determine this.")
    assert score == 0.5 and matched is False


def test_parse_empty_is_unmatched_half():
    score, matched = parse_judgment("")
    assert score == 0.5 and matched is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_codecheck_prompt_score.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Implement `build_judge_prompt` and `parse_judgment`**

Create `codecheck/prompt_score.py`:

```python
from __future__ import annotations
import re

JUDGE_TEMPLATE = (
    "Implementation:\n{main_code}\n\n"
    "Does the following construct from another implementation have behavior "
    "consistent with the implementation above?\n"
    "Construct:\n{sample_code}\n\n"
    "Answer Yes / No / N/A with a one-sentence justification."
)

# Map a matched answer token to an inconsistency score (higher = more likely incorrect).
_ANSWER = re.compile(r"\b(yes|no|n/?a)\b", re.IGNORECASE)


def build_judge_prompt(main_code: str, sample_code: str) -> str:
    return JUDGE_TEMPLATE.format(main_code=main_code, sample_code=sample_code)


def parse_judgment(text: str | None) -> tuple[float, bool]:
    """(inconsistency_score, matched). Yes->0.0, No->1.0, N/A->0.5.
    Unparseable / empty -> (0.5, False) so callers can count parse failures."""
    if not text:
        return 0.5, False
    m = _ANSWER.search(text)
    if not m:
        return 0.5, False
    tok = m.group(1).lower().replace("/", "")
    if tok == "yes":
        return 0.0, True
    if tok == "no":
        return 1.0, True
    return 0.5, True   # "na"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_codecheck_prompt_score.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add codecheck/prompt_score.py tests/test_codecheck_prompt_score.py
git commit -m "feat(codecheck): judge prompt template + Yes/No/N-A parsing"
```

---

## Task 4: `PromptJudge` — concurrent judging + aggregation

**Files:**
- Modify: `codecheck/prompt_score.py`
- Test: `tests/test_codecheck_prompt_score.py`

`PromptJudge.score(main_code, sample_codes)` fires one judge call per sample concurrently (mirroring `CodeGenerator`), parses each, and returns the mean inconsistency. It accumulates `parse_failures` so the run can report the judge parse-failure rate (an iteration-2 feedback item).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_codecheck_prompt_score.py`:

```python
from types import SimpleNamespace
from codecheck.prompt_score import PromptJudge


class FakeJudgeClient:
    """Returns a queued answer per call, in call order."""
    def __init__(self, answers):
        self._answers = list(answers)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._answers[len(self.calls) - 1]
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_judge_score_is_mean_inconsistency():
    client = FakeJudgeClient(["Yes.", "No.", "Yes."])  # 0.0, 1.0, 0.0 -> mean 1/3
    judge = PromptJudge(client, model="m")
    score = judge.score("def f(): return 1", ["a", "b", "c"])
    assert abs(score - (1.0 / 3.0)) < 1e-9
    assert judge.parse_failures == 0


def test_judge_counts_parse_failures():
    client = FakeJudgeClient(["Yes.", "uhh dunno"])     # second is unparseable
    judge = PromptJudge(client, model="m")
    judge.score("main", ["a", "b"])
    assert judge.parse_failures == 1


def test_judge_empty_samples_scores_zero():
    client = FakeJudgeClient([])
    judge = PromptJudge(client, model="m")
    assert judge.score("main", []) == 0.0


def test_judge_disables_reasoning_by_default():
    client = FakeJudgeClient(["Yes."])
    PromptJudge(client, model="m").score("main", ["a"])
    assert client.calls[0]["extra_body"] == {"reasoning": {"enabled": False}}
    assert client.calls[0]["temperature"] == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_codecheck_prompt_score.py -q`
Expected: FAIL (`PromptJudge` undefined).

- [ ] **Step 3: Implement `PromptJudge`**

Append to `codecheck/prompt_score.py` (add `from concurrent.futures import ThreadPoolExecutor` to the imports at the top):

```python
class PromptJudge:
    """LLM-as-judge consistency scorer. score() returns mean inconsistency
    over the samples; parse_failures accumulates unparseable judgments."""

    def __init__(self, client, model: str, think: bool = False, max_workers: int | None = None) -> None:
        self.client = client
        self.model = model
        self.think = think
        self.max_workers = max_workers
        self.parse_failures = 0

    def _judge_one(self, main_code: str, sample_code: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": build_judge_prompt(main_code, sample_code)}],
            temperature=0.0,
            extra_body={"reasoning": {"enabled": self.think}},
        )
        if not resp.choices:
            return ""
        return resp.choices[0].message.content or ""

    def score(self, main_code: str, sample_codes: list[str]) -> float:
        if not sample_codes:
            return 0.0
        with ThreadPoolExecutor(max_workers=self.max_workers or len(sample_codes)) as ex:
            raws = list(ex.map(lambda s: self._judge_one(main_code, s), sample_codes))
        values = []
        for raw in raws:
            value, matched = parse_judgment(raw)
            if not matched:
                self.parse_failures += 1
            values.append(value)
        return sum(values) / len(values)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_codecheck_prompt_score.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add codecheck/prompt_score.py tests/test_codecheck_prompt_score.py
git commit -m "feat(codecheck): PromptJudge concurrent judging + aggregation"
```

---

## Task 5: Multi-method pipeline

**Files:**
- Modify: `codecheck/pipeline.py`
- Test: `tests/test_codecheck_pipeline.py`

`score_problem` keeps generating + executing + labeling once, then fills the `scores` dict for every requested method. Exec reuses the already-computed `main_outputs`/`sample_outputs`; Prompt calls the judge on the codes. Existing pipeline tests that read `result.exec_score` keep working via the property, but the call now passes a `methods` set.

- [ ] **Step 1: Update/extend the failing tests**

In `tests/test_codecheck_pipeline.py`, find the call(s) to `score_problem` and pass `methods={"exec"}`. Then append a multi-method test (uses the existing fake generator/harness fixtures in that file; if they are named differently, reuse whatever the file already defines):

```python
from codecheck.prompt_score import PromptJudge
from tests.test_codecheck_prompt_score import FakeJudgeClient


def test_score_problem_fills_exec_and_prompt(monkeypatch):
    # Reuse the module's existing fake generator + harness used by the exec test.
    problem, generator, harness = _make_exec_fixture()   # replace with this file's existing helper
    judge = PromptJudge(FakeJudgeClient(["No.", "No."]), model="m")  # both inconsistent -> 1.0
    result = score_problem(problem, generator, harness, n_samples=2, timeout=1.0,
                           methods={"exec", "prompt"}, judge=judge)
    assert "exec" in result.scores and "prompt" in result.scores
    assert result.scores["prompt"] == 1.0
```

> If `tests/test_codecheck_pipeline.py` has no reusable fixture helper, build the problem/generator/harness inline the same way the file's existing exec test does — do not invent new harness behavior.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_codecheck_pipeline.py -q`
Expected: FAIL (`score_problem` has no `methods`/`judge` params; result has `exec_score` field not `scores`).

- [ ] **Step 3: Rewrite `score_problem` and `run_dataset`**

In `codecheck/pipeline.py` replace `score_problem` and `run_dataset` with:

```python
def score_problem(problem, generator, harness, n_samples: int, timeout: float = 5.0,
                  methods: set[str] | None = None, judge=None) -> CodeResult:
    methods = methods or {"exec"}
    main_code, sample_codes = generator.generate(problem, n_samples)
    main_outputs = _run_vector(main_code, problem, harness, timeout)
    sample_outputs = [_run_vector(code, problem, harness, timeout) for code in sample_codes]
    expected = expected_outputs(problem, harness, timeout)

    scores: dict[str, float] = {}
    if "exec" in methods:
        scores["exec"] = exec_inconsistency(main_outputs, sample_outputs)
    if "prompt" in methods:
        if judge is None:
            raise ValueError("method 'prompt' requires a judge")
        scores["prompt"] = judge.score(main_code, sample_codes)

    return CodeResult(
        task_id=problem.task_id,
        scores=scores,
        is_correct=is_correct(main_outputs, expected),
        main_code=main_code,
        sample_codes=sample_codes,
    )


def run_dataset(problems, generator, harness, n_samples: int, timeout: float = 5.0,
                methods: set[str] | None = None, judge=None) -> list[CodeResult]:
    return [score_problem(p, generator, harness, n_samples, timeout, methods, judge)
            for p in tqdm(problems, desc="codecheck")]
```

- [ ] **Step 4: Run the full suite**

Run: `pytest tests/ -q`
Expected: PASS (fix any other call sites still passing a positional `exec_score` or omitting `methods`).

- [ ] **Step 5: Commit**

```bash
git add codecheck/pipeline.py tests/test_codecheck_pipeline.py
git commit -m "feat(codecheck): score_problem fills scores for selected methods"
```

---

## Task 6: CLI `run --method` + judge wiring

**Files:**
- Modify: `run_codecheck.py`
- Test: `tests/test_codecheck_cli.py`

`--method {exec,prompt,both}` selects which scorers run. The judge reuses the same OpenAI client and model as generation. After a run that used the judge, print the parse-failure count.

- [ ] **Step 1: Write the failing test**

In `tests/test_codecheck_cli.py`, add a parser-level test (match how the file already tests `build_parser`):

```python
def test_run_parser_accepts_method():
    from run_codecheck import build_parser
    args = build_parser().parse_args(["run", "--method", "both", "--limit", "2"])
    assert args.method == "both"


def test_run_parser_method_defaults_to_exec():
    from run_codecheck import build_parser
    args = build_parser().parse_args(["run", "--limit", "2"])
    assert args.method == "exec"


def test_run_parser_rejects_unknown_method():
    import pytest
    from run_codecheck import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--method", "ast"])
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_codecheck_cli.py -q`
Expected: FAIL (no `--method`).

- [ ] **Step 3: Add `--method` to the parser**

In `run_codecheck.py`, inside `build_parser`, add to the `run_p` block (after `--think`):

```python
    run_p.add_argument("--method", choices=["exec", "prompt", "both"], default="exec",
                       help="which consistency scorer(s) to run")
```

- [ ] **Step 4: Wire the judge into `_cmd_run`**

Replace the body of `_cmd_run` in `run_codecheck.py` with:

```python
def _cmd_run(args: argparse.Namespace) -> None:
    from openai import AuthenticationError, OpenAI
    from codecheck.dataset import load_mbpp_plus
    from codecheck.generation import CodeGenerator
    from codecheck.prompt_score import PromptJudge
    from codecheck.execution import run_batch_in_subprocess
    from codecheck.pipeline import run_dataset, save_results

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit("error: missing OPENROUTER_API_KEY (see .env.example)")
    model = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.5-9b").strip()
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=0)
    generator = CodeGenerator(client, model=model, think=args.think)

    methods = {"exec", "prompt"} if args.method == "both" else {args.method}
    judge = PromptJudge(client, model=model, think=args.think) if "prompt" in methods else None

    problems = load_mbpp_plus(limit=args.limit, randomize=args.randomize, seed=args.seed)
    try:
        results = run_dataset(problems, generator, run_batch_in_subprocess,
                              n_samples=args.n, timeout=args.timeout,
                              methods=methods, judge=judge)
    except AuthenticationError:
        sys.exit("error: OpenRouter rejected OPENROUTER_API_KEY (expects an sk-or-v1-… key; see .env.example)")
    save_results(results, args.output)
    print(f"Saved {len(results)} results to {args.output}")
    if judge is not None:
        print(f"Judge parse failures: {judge.parse_failures}")
```

- [ ] **Step 5: Run the suite**

Run: `pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add run_codecheck.py tests/test_codecheck_cli.py
git commit -m "feat(codecheck): run --method selects exec/prompt/both"
```

---

## Task 7: CLI `evaluate` — per-method comparison readout

**Files:**
- Modify: `run_codecheck.py`
- Test: `tests/test_codecheck_cli.py` (or `tests/test_evaluate_cli.py` if CLI eval tests live there)

`evaluate` should detect every method present in the results and print, for each: AUC-PR (trapezoidal), the prevalence baseline, and the per-class histogram. This is the Exec-vs-Prompt side-by-side the iteration exists to produce.

- [ ] **Step 1: Write the failing test (a pure formatter, so it is unit-testable)**

Add to the CLI test file:

```python
def test_format_evaluation_lists_each_method():
    from run_codecheck import format_evaluation
    from codecheck.models import CodeResult
    results = [
        CodeResult("a", {"exec": 0.9, "prompt": 0.2}, False, "m", []),
        CodeResult("b", {"exec": 0.1, "prompt": 0.8}, True, "m", []),
    ]
    text = format_evaluation(results)
    assert "exec" in text and "prompt" in text
    assert "AUC-PR" in text
    assert "baseline" in text.lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_codecheck_cli.py -q`
Expected: FAIL (`format_evaluation` undefined).

- [ ] **Step 3: Add `format_evaluation` and use it in `_cmd_evaluate`**

In `run_codecheck.py`, add this helper (top-level) and rewrite `_cmd_evaluate`:

```python
def _methods_present(results) -> list[str]:
    seen: list[str] = []
    for r in results:
        for m in r.scores:
            if m not in seen:
                seen.append(m)
    return seen


def format_evaluation(results) -> str:
    from codecheck.evaluate import (
        auc_pr_detect_incorrect, prevalence_baseline, score_histogram,
    )
    n = len(results)
    n_incorrect = sum(1 for r in results if not r.is_correct)
    lines = [f"n={n}  incorrect={n_incorrect}  correct={n - n_incorrect}",
             f"baseline (incorrect prevalence): {prevalence_baseline(results):.4f}", ""]
    for method in _methods_present(results):
        auc_pr = auc_pr_detect_incorrect(results, method=method)
        lines.append(f"[{method}] AUC-PR (detect incorrect): {auc_pr:.4f}")
        lines.append(f"  {method} score histogram (bin: correct/incorrect):")
        for b in score_histogram(results, method=method, bins=10):
            lines.append(f"    [{b['lo']:.1f},{b['hi']:.1f}) {b['correct']}/{b['incorrect']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from codecheck.pipeline import load_results

    try:
        results = load_results(args.results)
    except FileNotFoundError:
        sys.exit(f"error: results file not found: {args.results}")
    print(format_evaluation(results))
```

- [ ] **Step 4: Run the suite**

Run: `pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add run_codecheck.py tests/test_codecheck_cli.py
git commit -m "feat(codecheck): evaluate prints per-method AUC-PR + baseline + histogram"
```

---

## Task 8: Docs — `--method` usage and standing run protocol

**Files:**
- Modify: `codecheck/README.md`

- [ ] **Step 1: Update the README**

Add a "SelfCheck-Prompt (iteration 2)" section documenting:

```markdown
## Methods

`run --method {exec,prompt,both}` selects the consistency scorer(s). A single run
generates each problem's implementations once and scores them with every selected
method, so Exec and Prompt are always compared on identical data.

- `exec` (default) — behavioral I/O divergence across samples.
- `prompt` — LLM-as-judge: per sample, is its behavior consistent with the main
  implementation? Yes→0.0 / No→1.0 / N-A→0.5, averaged over the N samples.
- `both` — runs both; `evaluate` then prints a per-method comparison.

The judge reuses `OPENROUTER_MODEL`. After a judged run the CLI prints the judge
parse-failure count.

### Standing run/report protocol (from iteration-1 findings)

- Sample problems **randomly across the full set** with a capable model — never the
  low-numbered slice (it is unrepresentatively easy and yields a single class).
- `evaluate` prints, per method, the AUC-PR, the **prevalence baseline** (the PR
  no-skill floor), and a **per-class score histogram**. Read AUC-PR against the
  baseline, not in isolation; the histogram exposes tie pile-ups at 0 that make
  the scalar fragile.
```

- [ ] **Step 2: Commit**

```bash
git add codecheck/README.md
git commit -m "docs(codecheck): document --method and standing run/report protocol"
```

---

## Manual verification (after all tasks; needs OPENROUTER_API_KEY)

Run the comparison on a real random sample and read it as the team will:

```bash
python run_codecheck.py run --method both --limit 30 --n 5 --seed 1 --timeout 5 \
  --output output/iter2-both.json
python run_codecheck.py evaluate --results output/iter2-both.json
```

Check:
- Both `exec` and `prompt` AUC-PR print, with the shared baseline.
- The histograms show each method's score spread per class.
- **The iteration's core question:** do any of the incorrect mains that Exec scores
  in bin `[0.0,0.1)` (its confident-consistent misses) get a high `prompt` score?
  That is the signal that the judge complements Exec.
- Note the judge parse-failure count and rough per-call latency (feeds iteration-4
  sizing).

Save a short report to `docs/reports/05-codecheck-iteration2-*.md` with the two
AUC-PRs, the baseline, the disagreeing rows, and the parse-failure rate.

---

## Self-Review (done while writing; recorded for the executor)

- **Spec coverage:** Prompt scorer (Tasks 3–4), Yes/No/N-A aggregation (Task 4),
  `--method` selection + same-data scoring (Tasks 5–6), Exec-vs-Prompt comparison
  readout (Task 7), parse-failure handling/counting (Tasks 3,4,6), baseline +
  histogram rigor (Task 2). All roadmap iteration-2 fields covered.
- **Type consistency:** `score(main_code, sample_codes) -> float` and
  `PromptJudge.parse_failures` used identically across Tasks 4–6;
  `score_histogram` bucket dict keys (`lo/hi/correct/incorrect`) match between
  Task 2 and Task 7; `scores` dict shape consistent from Task 1 onward.
- **Open item carried, not invented:** unit granularity (per-problem vs per-sample)
  stays an OPEN methods-doc decision, not silently chosen here.
```
