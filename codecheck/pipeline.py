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
    # harness is a batch harness: one call runs all inputs (one spawn per impl).
    outcomes = harness(code, problem.entry_point, problem.inputs, timeout)
    return [normalize_output(o, problem.atol) for o in outcomes]


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
        n_inputs=len(problem.inputs),
    )


def run_dataset(problems, generator, harness, n_samples: int, timeout: float = 5.0,
                methods: set[str] | None = None, judge=None) -> list[CodeResult]:
    return [score_problem(p, generator, harness, n_samples, timeout, methods, judge)
            for p in tqdm(problems, desc="codecheck")]


def save_results(results: list[CodeResult], path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp", encoding="utf-8") as tmp:
        json.dump([r.to_dict() for r in results], tmp, indent=2, ensure_ascii=False)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def load_results(path: str | os.PathLike) -> list[CodeResult]:
    with open(path, encoding="utf-8") as f:
        return [CodeResult.from_dict(item) for item in json.load(f)]
