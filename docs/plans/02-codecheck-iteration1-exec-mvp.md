# SelfCheck-Exec MVP (Iteration 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **On approval, before Task 1:** save this plan to `docs/plans/02-codecheck-iteration1-exec-mvp.md` (plan mode could not write there). It is iteration 1 of `docs/plans/01-codecheck-roadmap.md`.

**Goal:** Stand up an end-to-end SelfCheck-Exec pipeline that, on a handful of MBPP+ problems, generates a main implementation (T=0) + N samples (T=1), scores the main implementation's behavioral-consistency against the samples, labels it correct/incorrect by execution, and reports AUC-PR.

**Architecture:** A new top-level `codecheck/` package mirroring the existing `replication/` layout. Pure, testable units (models, exec scoring, labeling, evaluate) are separated from side-effecting units (subprocess execution harness, OpenRouter generation). A thin CLI `run_codecheck.py` mirrors `score.py`. Ground-truth correctness is computed by running each problem's canonical solution through the same execution harness, so no human annotation and no dependency on evalplus's expected-output API.

**Tech Stack:** Python, `evalplus` (MBPP+ problems + structured inputs), `openai` SDK against OpenRouter (reusing the existing client pattern), `multiprocessing` (spawn) for the sandbox, `scikit-learn` `precision_recall_curve`/`auc` for AUC-PR (same as `replication/evaluation/metrics.py`), `pytest`.

---

## Context

`docs/source/04-codecheck-methods.md` defines three code-domain SelfCheck variants; the roadmap (`docs/plans/01-codecheck-roadmap.md`) makes **SelfCheck-Exec on MBPP+** iteration 1 because the execution harness it forces doubles as the ground-truth labeler. Confirmed forks: MVP method = Exec; dataset = MBPP+ via **evalplus** (structured input lists, no assert parsing); code-generation pipeline **built fresh** (the WikiBio replication shipped pre-generated samples — no reusable generator exists). This iteration must *run* and produce a real consistency-vs-correctness readout; signal quality and the "models too smart on MBPP+" question are explicit feedback to collect, not blockers.

## Reused existing code (do not reinvent)

- **AUC-PR**: `replication/evaluation/metrics.py:38-41` uses `precision_recall_curve` + `auc` (trapezoidal). Mirror exactly.
- **Atomic JSON save**: `replication/score/base.py:46-54` (`ScoreIO.save_results`) — temp file + `os.replace`, `ensure_ascii=False`. Mirror in `pipeline.save_results`.
- **Env loading**: `score.py:22-41` (`load_environment`, `_load_env_file`). Import and reuse in the CLI.
- **OpenRouter client construction**: `selfcheckgpt/modeling_selfcheck_apiprompt.py:46-52` — `OpenAI(base_url, api_key, timeout=60, max_retries=0)` and the empty-`choices` guard pattern. Reuse the construction shape; env vars `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` / `OPENROUTER_BASE_URL` per `replication/score/prompt.py:26-36` (default model `qwen/qwen3.5-9b`).
- **Dataclass + `from_dict`/`to_dict` entity pattern**: `replication/entity/passage_result.py`.

## File structure

```
codecheck/
  __init__.py
  models.py        # CodeProblem, CodeResult dataclasses (+ to/from dict)
  execution.py     # run_in_subprocess (sandbox), normalize_output, _canonical
  exec_score.py    # exec_inconsistency (pure)
  labeling.py      # expected_outputs, is_correct
  dataset.py       # load_mbpp_plus -> list[CodeProblem], cache to data/mbpp_plus.json
  generation.py    # CodeGenerator (injectable client), extract_code, build_prompt
  pipeline.py      # score_problem, run_dataset, save_results, load_results
  evaluate.py      # auc_pr_detect_incorrect
run_codecheck.py   # CLI: `run` (generate+score+save) and `evaluate`
tests/
  test_codecheck_models.py
  test_codecheck_execution.py
  test_codecheck_exec_score.py
  test_codecheck_labeling.py
  test_codecheck_dataset.py
  test_codecheck_generation.py
  test_codecheck_pipeline.py
  test_codecheck_evaluate.py
  test_codecheck_cli.py
```

`requirements.txt`: add `evalplus`.

Data shapes (the contract every task shares):
- `CodeProblem(task_id: str, prompt: str, entry_point: str, canonical_solution: str, inputs: list[list], atol: float)` — `inputs` is a list of argument-lists; `harness(code, entry_point, args, timeout)` calls `entry_point(*args)`.
- Normalized output = a hashable tuple: `("value", canonical)` for a return value, or `("status", "__TIMEOUT__" | "__ERROR__")` for a failed run. Equality of normalized outputs is the consistency primitive.
- `CodeResult(task_id: str, exec_score: float, is_correct: bool, main_code: str, sample_codes: list[str])`.

---

## Task 1: Data models

**Files:**
- Create: `codecheck/__init__.py` (empty)
- Create: `codecheck/models.py`
- Test: `tests/test_codecheck_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_codecheck_models.py
from codecheck.models import CodeProblem, CodeResult


def test_code_problem_roundtrip():
    p = CodeProblem(
        task_id="Mbpp/2", prompt="def f(x):\n    'doc'\n", entry_point="f",
        canonical_solution="def f(x):\n    return x + 1\n",
        inputs=[[1], [2]], atol=1e-6,
    )
    assert CodeProblem.from_dict(p.to_dict()) == p


def test_code_result_roundtrip():
    r = CodeResult(task_id="Mbpp/2", exec_score=0.4, is_correct=True,
                   main_code="def f(x): return x", sample_codes=["def f(x): return x", "def f(x): return 0"])
    assert CodeResult.from_dict(r.to_dict()) == r
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_codecheck_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'codecheck'`.

- [ ] **Step 3: Write minimal implementation**

```python
# codecheck/models.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CodeProblem:
    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str
    inputs: list[list]
    atol: float = 1e-6

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "prompt": self.prompt, "entry_point": self.entry_point,
            "canonical_solution": self.canonical_solution, "inputs": self.inputs, "atol": self.atol,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CodeProblem":
        return cls(
            task_id=d["task_id"], prompt=d["prompt"], entry_point=d["entry_point"],
            canonical_solution=d["canonical_solution"], inputs=d["inputs"], atol=d.get("atol", 1e-6),
        )


@dataclass
class CodeResult:
    task_id: str
    exec_score: float
    is_correct: bool
    main_code: str
    sample_codes: list[str]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "exec_score": self.exec_score, "is_correct": self.is_correct,
            "main_code": self.main_code, "sample_codes": self.sample_codes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CodeResult":
        return cls(
            task_id=d["task_id"], exec_score=d["exec_score"], is_correct=d["is_correct"],
            main_code=d["main_code"], sample_codes=d["sample_codes"],
        )
```

Create empty `codecheck/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_codecheck_models.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add codecheck/__init__.py codecheck/models.py tests/test_codecheck_models.py
git commit -m "feat(codecheck): add CodeProblem/CodeResult models"
```

---

## Task 2: Execution sandbox + output normalization

**Files:**
- Create: `codecheck/execution.py`
- Test: `tests/test_codecheck_execution.py`

The sandbox runs untrusted, model-generated code. Each call runs in a **fresh spawned process** joined with a timeout and force-terminated on overrun (so infinite loops cannot hang the run). Output is normalized to a hashable, tolerance-bucketed form so equality is the consistency primitive.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_codecheck_execution.py
from codecheck.execution import run_in_subprocess, normalize_output

ADD = "def f(x):\n    return x + 1\n"
BOOM = "def f(x):\n    raise ValueError('boom')\n"
HANG = "def f(x):\n    while True:\n        pass\n"
FLOATY = "def f(x):\n    return 0.1 + 0.2\n"


def test_runs_and_returns_value():
    assert run_in_subprocess(ADD, "f", [1], timeout=5.0) == ("ok", 2)


def test_exception_becomes_error_status():
    status, _ = run_in_subprocess(BOOM, "f", [1], timeout=5.0)
    assert status == "err"


def test_timeout_is_killed():
    status, _ = run_in_subprocess(HANG, "f", [1], timeout=1.0)
    assert status == "timeout"


def test_normalize_equality_and_float_tolerance():
    a = normalize_output(run_in_subprocess(FLOATY, "f", [0], timeout=5.0), atol=1e-6)
    b = normalize_output(("ok", 0.3), atol=1e-6)
    assert a == b                       # 0.1+0.2 ≈ 0.3 within atol
    assert normalize_output(("err", None)) == normalize_output(("err", None))
    assert normalize_output(("ok", 2)) != normalize_output(("timeout", None))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_codecheck_execution.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'codecheck.execution'`.

- [ ] **Step 3: Write minimal implementation**

```python
# codecheck/execution.py
from __future__ import annotations
import multiprocessing as mp


def _worker(code: str, entry_point: str, args: list, q) -> None:
    try:
        ns: dict = {}
        exec(code, ns)
        result = ns[entry_point](*args)
        q.put(("ok", result))
    except Exception as exc:  # noqa: BLE001 — any failure of untrusted code is an "err"
        q.put(("err", repr(exc)))


def run_in_subprocess(code: str, entry_point: str, args: list, timeout: float = 5.0):
    """Run entry_point(*args) defined in `code` in a fresh process.

    Returns one of: ("ok", value) | ("err", repr) | ("timeout", None).
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(code, entry_point, list(args), q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return ("timeout", None)
    try:
        return q.get_nowait()
    except Exception:
        return ("err", None)


def _canonical(value, atol: float):
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value / atol) if atol else value
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(v, atol) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_canonical(v, atol) for v in value))
    if isinstance(value, dict):
        return tuple(sorted((k, _canonical(v, atol)) for k, v in value.items()))
    return value


def normalize_output(outcome, atol: float = 1e-6):
    """Map a run outcome to a hashable, comparable form."""
    status, value = outcome
    if status != "ok":
        return ("status", status)
    return ("value", _canonical(value, atol))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_codecheck_execution.py -v`
Expected: PASS (4 passed). (The timeout test takes ~1s.)

- [ ] **Step 5: Commit**

```bash
git add codecheck/execution.py tests/test_codecheck_execution.py
git commit -m "feat(codecheck): add subprocess sandbox and output normalization"
```

---

## Task 3: Exec inconsistency score (pure)

**Files:**
- Create: `codecheck/exec_score.py`
- Test: `tests/test_codecheck_exec_score.py`

Score = `1 - mean agreement` between the main implementation's normalized output vector and each sample's, over the shared inputs. `1.0` ⇒ fully inconsistent ⇒ likely hallucinated (matches SelfCheck's "1 = hallucinated" convention).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_codecheck_exec_score.py
from codecheck.exec_score import exec_inconsistency

MAIN = [("value", 1), ("value", 2), ("value", 3)]


def test_all_samples_agree_is_zero():
    assert exec_inconsistency(MAIN, [MAIN, MAIN]) == 0.0


def test_all_samples_disagree_is_one():
    other = [("value", 9), ("value", 9), ("value", 9)]
    assert exec_inconsistency(MAIN, [other, other]) == 1.0


def test_partial_agreement():
    half = [("value", 1), ("value", 2), ("value", 9)]  # 2/3 match
    assert exec_inconsistency(MAIN, [half]) == 1.0 - (2 / 3)


def test_no_samples_is_zero():
    assert exec_inconsistency(MAIN, []) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_codecheck_exec_score.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# codecheck/exec_score.py
from __future__ import annotations


def exec_inconsistency(main_outputs: list, sample_outputs: list[list]) -> float:
    """1 - mean per-sample agreement with main, over shared inputs. Range [0, 1]."""
    if not sample_outputs:
        return 0.0
    n_inputs = len(main_outputs)
    if n_inputs == 0:
        return 0.0
    agreements = []
    for sample in sample_outputs:
        matches = sum(1 for a, b in zip(main_outputs, sample) if a == b)
        agreements.append(matches / n_inputs)
    return 1.0 - sum(agreements) / len(agreements)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_codecheck_exec_score.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add codecheck/exec_score.py tests/test_codecheck_exec_score.py
git commit -m "feat(codecheck): add exec inconsistency score"
```

---

## Task 4: Ground-truth labeling

**Files:**
- Create: `codecheck/labeling.py`
- Test: `tests/test_codecheck_labeling.py`

The main implementation is **correct** iff it reproduces the canonical solution's normalized output on every input. Expected outputs come from running `canonical_solution` through the same harness — no external answer key needed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_codecheck_labeling.py
from codecheck.models import CodeProblem
from codecheck.execution import run_in_subprocess, normalize_output
from codecheck.labeling import expected_outputs, is_correct

PROBLEM = CodeProblem(
    task_id="t", prompt="", entry_point="f",
    canonical_solution="def f(x):\n    return x + 1\n",
    inputs=[[1], [2], [3]], atol=1e-6,
)


def _norm_run(code):
    return [normalize_output(run_in_subprocess(code, "f", args, 5.0), PROBLEM.atol) for args in PROBLEM.inputs]


def test_correct_when_matches_canonical():
    expected = expected_outputs(PROBLEM, run_in_subprocess)
    assert is_correct(_norm_run("def f(x):\n    return 1 + x\n"), expected) is True


def test_incorrect_when_diverges():
    expected = expected_outputs(PROBLEM, run_in_subprocess)
    assert is_correct(_norm_run("def f(x):\n    return x\n"), expected) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_codecheck_labeling.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# codecheck/labeling.py
from __future__ import annotations
from codecheck.execution import normalize_output


def expected_outputs(problem, harness, timeout: float = 5.0) -> list:
    """Normalized outputs of the canonical solution over the problem's inputs."""
    return [
        normalize_output(harness(problem.canonical_solution, problem.entry_point, args, timeout), problem.atol)
        for args in problem.inputs
    ]


def is_correct(main_outputs: list, expected: list) -> bool:
    return len(main_outputs) == len(expected) and all(a == b for a, b in zip(main_outputs, expected))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_codecheck_labeling.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add codecheck/labeling.py tests/test_codecheck_labeling.py
git commit -m "feat(codecheck): add execution-based ground-truth labeling"
```

---

## Task 5: MBPP+ dataset loader (evalplus)

**Files:**
- Create: `codecheck/dataset.py`
- Test: `tests/test_codecheck_dataset.py`

Loads MBPP+ via `evalplus.data.get_mbpp_plus()`, maps each problem to `CodeProblem` (inputs = `base_input` + `plus_input`), and caches to `data/mbpp_plus.json` so later runs and offline tests don't re-fetch.

- [ ] **Step 1: Verify the evalplus schema (one-off, informs the mapping)**

Run: `python -c "from evalplus.data import get_mbpp_plus; p=next(iter(get_mbpp_plus().values())); print(sorted(p.keys()))"`
Expected: keys including `task_id`, `prompt`, `entry_point`, `canonical_solution`, `base_input`, `plus_input`, `atol`. If a name differs, adjust the field mapping in Step 3 accordingly (only the dict keys change).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_codecheck_dataset.py
import codecheck.dataset as ds
from codecheck.models import CodeProblem

FAKE = {
    "Mbpp/2": {
        "task_id": "Mbpp/2", "prompt": "def f(x):\n    'doc'\n", "entry_point": "f",
        "canonical_solution": "def f(x):\n    return x + 1\n",
        "base_input": [[1], [2]], "plus_input": [[3]], "atol": 0.0,
    }
}


def test_maps_evalplus_to_code_problems(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "get_mbpp_plus", lambda: FAKE)
    cache = tmp_path / "mbpp_plus.json"
    problems = ds.load_mbpp_plus(limit=1, cache_path=cache)
    assert problems == [CodeProblem(
        task_id="Mbpp/2", prompt="def f(x):\n    'doc'\n", entry_point="f",
        canonical_solution="def f(x):\n    return x + 1\n",
        inputs=[[1], [2], [3]], atol=0.0,
    )]
    assert cache.exists()  # cached for reuse
```

- [ ] **Step 3: Write minimal implementation**

```python
# codecheck/dataset.py
from __future__ import annotations
import json
from pathlib import Path

from evalplus.data import get_mbpp_plus

from codecheck.models import CodeProblem

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "data" / "mbpp_plus.json"


def _to_problem(item: dict) -> CodeProblem:
    return CodeProblem(
        task_id=item["task_id"],
        prompt=item["prompt"],
        entry_point=item["entry_point"],
        canonical_solution=item["canonical_solution"],
        inputs=list(item.get("base_input", [])) + list(item.get("plus_input", [])),
        atol=item.get("atol", 1e-6),
    )


def load_mbpp_plus(limit: int | None = None, cache_path: Path = DEFAULT_CACHE) -> list[CodeProblem]:
    problems = [_to_problem(item) for item in get_mbpp_plus().values()]
    if limit is not None:
        problems = problems[:limit]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps([p.to_dict() for p in problems], indent=2), encoding="utf-8")
    return problems
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_codecheck_dataset.py -v`
Expected: PASS (1 passed). (Test monkeypatches `get_mbpp_plus`, so no network.)

- [ ] **Step 5: Commit**

```bash
git add codecheck/dataset.py tests/test_codecheck_dataset.py
git commit -m "feat(codecheck): add MBPP+ loader via evalplus with caching"
```

---

## Task 6: Code generation (OpenRouter)

**Files:**
- Create: `codecheck/generation.py`
- Test: `tests/test_codecheck_generation.py`

`CodeGenerator` takes an injected client (so tests never hit the API), generates the main implementation at `temperature=0.0` and `n_samples` at `temperature=1.0`, and extracts code from markdown fences.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_codecheck_generation.py
from types import SimpleNamespace
from codecheck.generation import CodeGenerator, extract_code
from codecheck.models import CodeProblem

PROBLEM = CodeProblem(task_id="t", prompt="def f(x):\n    'add one'\n", entry_point="f",
                      canonical_solution="def f(x):\n    return x+1\n", inputs=[[1]], atol=0.0)


def test_extract_code_from_fence():
    assert extract_code("blah\n```python\ndef f(x):\n    return x\n```\n") == "def f(x):\n    return x"
    assert extract_code("def f(x):\n    return x") == "def f(x):\n    return x"


class FakeClient:
    def __init__(self):
        self.calls = []
        content = "```python\ndef f(x):\n    return x + 1\n```"
        msg = SimpleNamespace(content=content)
        self._resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._resp


def test_generate_uses_temperatures_0_and_1():
    client = FakeClient()
    gen = CodeGenerator(client, model="m")
    main, samples = gen.generate(PROBLEM, n_samples=3)
    assert main == "def f(x):\n    return x + 1"
    assert len(samples) == 3
    temps = [c["temperature"] for c in client.calls]
    assert temps == [0.0, 1.0, 1.0, 1.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_codecheck_generation.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# codecheck/generation.py
from __future__ import annotations
import re

_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)

PROMPT_TEMPLATE = (
    "Complete the following Python function. "
    "Return only the complete function implementation, no explanation.\n\n{prompt}"
)


def extract_code(text: str | None) -> str:
    if not text:
        return ""
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()


def build_prompt(problem) -> str:
    return PROMPT_TEMPLATE.format(prompt=problem.prompt)


class CodeGenerator:
    def __init__(self, client, model: str) -> None:
        self.client = client
        self.model = model

    def _complete(self, prompt: str, temperature: float) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        if not resp.choices:
            return ""
        return extract_code(resp.choices[0].message.content)

    def generate(self, problem, n_samples: int) -> tuple[str, list[str]]:
        prompt = build_prompt(problem)
        main = self._complete(prompt, 0.0)
        samples = [self._complete(prompt, 1.0) for _ in range(n_samples)]
        return main, samples
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_codecheck_generation.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add codecheck/generation.py tests/test_codecheck_generation.py
git commit -m "feat(codecheck): add OpenRouter code generator (main T=0, samples T=1)"
```

---

## Task 7: Pipeline orchestration + result IO

**Files:**
- Create: `codecheck/pipeline.py`
- Test: `tests/test_codecheck_pipeline.py`

`score_problem` ties generation → execution → exec score → label into one `CodeResult`. `run_dataset` loops with a progress bar; `save_results`/`load_results` mirror the atomic-write pattern from `replication/score/base.py:46-54`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_codecheck_pipeline.py
from codecheck.models import CodeProblem, CodeResult
from codecheck.execution import run_in_subprocess
from codecheck.pipeline import score_problem, save_results, load_results

PROBLEM = CodeProblem(task_id="t", prompt="", entry_point="f",
                      canonical_solution="def f(x):\n    return x + 1\n",
                      inputs=[[1], [2], [3]], atol=0.0)


class StubGen:
    def __init__(self, main, samples):
        self._main, self._samples = main, samples

    def generate(self, problem, n_samples):
        return self._main, self._samples


def test_correct_main_with_consistent_samples():
    gen = StubGen("def f(x):\n    return x + 1\n",
                  ["def f(x):\n    return x + 1\n", "def f(x):\n    return 1 + x\n"])
    res = score_problem(PROBLEM, gen, run_in_subprocess, n_samples=2, timeout=5.0)
    assert res.is_correct is True
    assert res.exec_score == 0.0


def test_incorrect_main_with_divergent_samples():
    gen = StubGen("def f(x):\n    return x\n",                       # wrong
                  ["def f(x):\n    return x + 1\n", "def f(x):\n    return x + 1\n"])  # disagree w/ main
    res = score_problem(PROBLEM, gen, run_in_subprocess, n_samples=2, timeout=5.0)
    assert res.is_correct is False
    assert res.exec_score == 1.0


def test_save_and_load_roundtrip(tmp_path):
    results = [CodeResult("t", 0.5, True, "m", ["s"])]
    path = tmp_path / "out.json"
    save_results(results, path)
    assert load_results(path) == results
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_codecheck_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# codecheck/pipeline.py
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path

from tqdm import tqdm

from codecheck.models import CodeProblem, CodeResult
from codecheck.execution import normalize_output
from codecheck.exec_score import exec_inconsistency
from codecheck.labeling import expected_outputs, is_correct


def _run_vector(code: str, problem: CodeProblem, harness, timeout: float) -> list:
    return [normalize_output(harness(code, problem.entry_point, args, timeout), problem.atol)
            for args in problem.inputs]


def score_problem(problem, generator, harness, n_samples: int, timeout: float = 5.0) -> CodeResult:
    main_code, sample_codes = generator.generate(problem, n_samples)
    main_outputs = _run_vector(main_code, problem, harness, timeout)
    sample_outputs = [_run_vector(code, problem, harness, timeout) for code in sample_codes]
    expected = expected_outputs(problem, harness, timeout)
    return CodeResult(
        task_id=problem.task_id,
        exec_score=exec_inconsistency(main_outputs, sample_outputs),
        is_correct=is_correct(main_outputs, expected),
        main_code=main_code,
        sample_codes=sample_codes,
    )


def run_dataset(problems, generator, harness, n_samples: int, timeout: float = 5.0) -> list[CodeResult]:
    return [score_problem(p, generator, harness, n_samples, timeout)
            for p in tqdm(problems, desc="codecheck exec")]


def save_results(results: list[CodeResult], path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp", encoding="utf-8") as tmp:
        json.dump([r.to_dict() for r in results], tmp, indent=2, ensure_ascii=False)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def load_results(path: str | os.PathLike) -> list[CodeResult]:
    with open(path) as f:
        return [CodeResult.from_dict(item) for item in json.load(f)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_codecheck_pipeline.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add codecheck/pipeline.py tests/test_codecheck_pipeline.py
git commit -m "feat(codecheck): add pipeline orchestration and result IO"
```

---

## Task 8: Evaluation (AUC-PR detect-incorrect)

**Files:**
- Create: `codecheck/evaluate.py`
- Test: `tests/test_codecheck_evaluate.py`

Mirrors `replication/evaluation/metrics.py:38-41` (`precision_recall_curve` + trapezoidal `auc`). Positive class = **incorrect** (hallucinated) implementation; score = `exec_score`. Returns `nan` if only one class is present (AUC-PR undefined) — expected early-feedback signal if MBPP+ is too easy.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_codecheck_evaluate.py
import math
from codecheck.models import CodeResult
from codecheck.evaluate import auc_pr_detect_incorrect


def _r(score, correct):
    return CodeResult("t", score, correct, "m", ["s"])


def test_perfect_separation_scores_one():
    results = [_r(0.9, False), _r(0.8, False), _r(0.1, True), _r(0.0, True)]
    assert auc_pr_detect_incorrect(results) == 1.0


def test_single_class_is_nan():
    results = [_r(0.5, True), _r(0.4, True)]
    assert math.isnan(auc_pr_detect_incorrect(results))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_codecheck_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# codecheck/evaluate.py
from __future__ import annotations
from sklearn.metrics import auc, precision_recall_curve

from codecheck.models import CodeResult


def auc_pr_detect_incorrect(results: list[CodeResult]) -> float:
    y_true = [0 if r.is_correct else 1 for r in results]
    scores = [r.exec_score for r in results]
    if len(set(y_true)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return float(auc(recall, precision))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_codecheck_evaluate.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add codecheck/evaluate.py tests/test_codecheck_evaluate.py
git commit -m "feat(codecheck): add AUC-PR detect-incorrect metric"
```

---

## Task 9: CLI (`run_codecheck.py`)

**Files:**
- Create: `run_codecheck.py`
- Test: `tests/test_codecheck_cli.py`

Two subcommands mirroring `score.py`/`evaluate.py`: `run` (load MBPP+ → build generator from env → score → save) and `evaluate` (load results → print AUC-PR + counts). Reuses `load_environment` from `score.py`. The test exercises `evaluate` against a fixture results file (no API, no evalplus).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_codecheck_cli.py
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_evaluate_subcommand_prints_auc(tmp_path):
    results = [
        {"task_id": "a", "exec_score": 0.9, "is_correct": False, "main_code": "m", "sample_codes": ["s"]},
        {"task_id": "b", "exec_score": 0.1, "is_correct": True, "main_code": "m", "sample_codes": ["s"]},
    ]
    path = tmp_path / "res.json"
    path.write_text(json.dumps(results))
    out = subprocess.run(
        [sys.executable, "run_codecheck.py", "evaluate", "--results", str(path)],
        cwd=REPO, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "AUC-PR" in out.stdout
    assert "n=2" in out.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_codecheck_cli.py -v`
Expected: FAIL — `run_codecheck.py` does not exist (non-zero return, assertion fails).

- [ ] **Step 3: Write minimal implementation**

```python
# run_codecheck.py
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from score import load_environment  # reuse existing env loader

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = REPO_ROOT / "output" / "codecheck-exec.json"


def _cmd_run(args: argparse.Namespace) -> None:
    from openai import OpenAI
    from codecheck.dataset import load_mbpp_plus
    from codecheck.generation import CodeGenerator
    from codecheck.execution import run_in_subprocess
    from codecheck.pipeline import run_dataset, save_results

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit("error: missing OPENROUTER_API_KEY (see .env.example)")
    model = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.5-9b").strip()
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=0)
    generator = CodeGenerator(client, model=model)

    problems = load_mbpp_plus(limit=args.limit)
    results = run_dataset(problems, generator, run_in_subprocess, n_samples=args.n, timeout=args.timeout)
    save_results(results, args.output)
    print(f"Saved {len(results)} results to {args.output}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from codecheck.pipeline import load_results
    from codecheck.evaluate import auc_pr_detect_incorrect

    results = load_results(args.results)
    n_incorrect = sum(1 for r in results if not r.is_correct)
    auc_pr = auc_pr_detect_incorrect(results)
    print(f"AUC-PR (detect incorrect): {auc_pr:.4f}")
    print(f"n={len(results)}  incorrect={n_incorrect}  correct={len(results) - n_incorrect}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SelfCheck-Exec on MBPP+ (iteration 1).")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="generate, score, and save")
    run_p.add_argument("--limit", type=int, default=20, help="number of MBPP+ problems")
    run_p.add_argument("--n", type=int, default=5, help="samples per problem (T=1)")
    run_p.add_argument("--timeout", type=float, default=5.0, help="per-call execution timeout (s)")
    run_p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="results JSON path")
    run_p.set_defaults(func=_cmd_run)

    eval_p = sub.add_parser("evaluate", help="report AUC-PR from a results file")
    eval_p.add_argument("--results", type=str, default=str(DEFAULT_OUTPUT), help="results JSON path")
    eval_p.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    load_environment()
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_codecheck_cli.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add run_codecheck.py tests/test_codecheck_cli.py
git commit -m "feat(codecheck): add run/evaluate CLI"
```

---

## Task 10: Dependency + full suite + roadmap status

**Files:**
- Modify: `requirements.txt` (add `evalplus`)
- Modify: `docs/plans/01-codecheck-roadmap.md` (mark iteration 1 status)

- [ ] **Step 1: Add evalplus**

Add `evalplus` to `requirements.txt` (after `datasets`). Then:
Run: `pip install evalplus`
Expected: installs without error.

- [ ] **Step 2: Run the whole codecheck suite**

Run: `pytest tests/test_codecheck_*.py -v`
Expected: all PASS (no API/network — generation and dataset are stubbed/monkeypatched).

- [ ] **Step 3: Mark roadmap iteration 1 in progress/landed**

In `docs/plans/01-codecheck-roadmap.md`, add a one-line status note under Iteration 1 (e.g. "Status: implemented, pending live MBPP+ run").

- [ ] **Step 4: Commit**

```bash
git add requirements.txt docs/plans/01-codecheck-roadmap.md
git commit -m "chore(codecheck): add evalplus dep; mark roadmap iteration 1"
```

---

## End-to-end verification (manual, live API)

This is the user test flow the roadmap calls for. Needs `OPENROUTER_API_KEY` in `.env`.

1. **Smoke run** on a few problems with small N:
   Run: `python run_codecheck.py run --limit 10 --n 5 --timeout 5`
   Expected: progress bar over 10 problems; `Saved 10 results to output/codecheck-exec.json`.
2. **Inspect** `output/codecheck-exec.json`: each entry has `exec_score` in `[0,1]`, an `is_correct` bool, the `main_code`, and `sample_codes`.
3. **Evaluate**:
   Run: `python run_codecheck.py evaluate --results output/codecheck-exec.json`
   Expected: a printed `AUC-PR (detect incorrect)` and the `n / incorrect / correct` counts.
4. **Judge the signal**: do high `exec_score` rows line up with `is_correct=False`? Open `data/mbpp_plus.json` to cross-check inputs.

## Feedback to collect (gates the next iterations)

- Does Exec inconsistency separate correct from incorrect? (go/no-go for the code direction)
- **Hallucination rate**: if `incorrect` ≈ 0 on MBPP+ (AUC-PR returns `nan`/unstable), the "models too smart" risk is real → pull roadmap iteration 5 (CodeHaluEval/Collu-Bench) earlier, and/or use a weaker generation model.
- Input set sufficiency: are `base_input + plus_input` enough to expose divergence, or do we need generated/fuzzed inputs?
- Output-equality edge cases surfaced in practice (floats/atol, unordered collections, both-error handling).
- Sensible `N` and per-call `timeout`.

## Risks / open decisions

- **Sandbox safety**: `exec` of model-generated code runs in a local spawned subprocess with a timeout — isolated enough to prevent hangs, but **not** containerized. Acceptable for a local course MVP; revisit (container/seccomp) before any untrusted-at-scale run.
- **Both-error equality**: two failing runs are treated as "consistent" only when they share the same status (`err`/`timeout`). If this inflates consistency for systematically-broken problems, revisit in iteration 4.
- **evalplus field names**: Task 5 Step 1 verifies the schema; if keys differ, only the mapping in `_to_problem` changes.

## Self-review notes

- Spec coverage: generation (T6), execution harness (T2), ground-truth labeling (T4), Exec scoring (T3), MBPP+ loading (T5), AUC-PR (T8), end-to-end run command (T9) — all roadmap iteration-1 deliverables covered.
- Type consistency: harness outcome `("ok"|"err"|"timeout", value)`; `normalize_output` → `("value"|"status", …)`; `CodeProblem.inputs: list[list]`; `harness(code, entry_point, args, timeout)` signature is identical across labeling/pipeline; `CodeResult` fields identical across pipeline/evaluate/CLI.
- No placeholders: every code step is complete and runnable.
