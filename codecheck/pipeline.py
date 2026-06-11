from __future__ import annotations
import json
import logging
import os
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

from codecheck.models import CodeProblem, CodeResult
from codecheck.execution import normalize_output
from codecheck.exec_score import exec_inconsistency
from codecheck.labeling import expected_outputs, is_correct

logger = logging.getLogger("codecheck.pipeline")


def _run_vector(code: str, problem: CodeProblem, harness, timeout: float) -> list:
    # harness is a batch harness: one call runs all inputs (one spawn per impl).
    outcomes = harness(code, problem.entry_point, problem.inputs, timeout)
    return [normalize_output(o, problem.atol) for o in outcomes]


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


def run_dataset(problems, generator, harness, n_samples: int, timeout: float = 5.0,
                methods: set[str] | None = None, judge=None, ast_scorer=None) -> list[CodeResult]:
    problems = list(problems)
    total = len(problems)
    results: list[CodeResult] = []
    failed: list[str] = []
    for i, problem in enumerate(tqdm(problems, desc="codecheck"), start=1):
        started = time.monotonic()
        try:
            result = score_problem(problem, generator, harness, n_samples, timeout, methods, judge, ast_scorer)
        except (KeyboardInterrupt, SystemExit):
            raise  # let the user abort the whole run
        except Exception:
            # One problem's failure (e.g. an exhausted-retry API timeout) must not abort
            # the run. Log the full traceback, skip it, and continue with the rest.
            logger.exception("problem %s failed; skipping", problem.task_id)
            failed.append(problem.task_id)
            continue
        elapsed = time.monotonic() - started
        scores = "  ".join(f"{name}={value:.3f}" for name, value in result.scores.items())
        # tqdm.write keeps these lines from corrupting the live progress bar.
        tqdm.write(f"[{i}/{total}] {result.task_id}  correct={result.is_correct}  "
                   f"{scores}  n_inputs={result.n_inputs}  ({elapsed:.1f}s)")
        results.append(result)
    if failed:
        logger.warning("%d/%d problems failed and were skipped: %s",
                       len(failed), total, ", ".join(failed))
    return results


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
