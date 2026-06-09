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
