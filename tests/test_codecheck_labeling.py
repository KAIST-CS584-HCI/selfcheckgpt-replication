from codecheck.models import CodeProblem
from codecheck.execution.sandbox import run_batch_in_subprocess, normalize_output
from codecheck.execution.labeling import expected_outputs, is_correct, has_error, count_outcomes

PROBLEM = CodeProblem(
    task_id="t", prompt="", entry_point="f",
    canonical_solution="def f(x):\n    return x + 1\n",
    inputs=[[1], [2], [3]], atol=1e-6,
)


def _norm_run(code):
    outcomes = run_batch_in_subprocess(code, "f", PROBLEM.inputs, 5.0)
    return [normalize_output(o, PROBLEM.atol) for o in outcomes]


def test_correct_when_matches_canonical():
    expected = expected_outputs(PROBLEM, run_batch_in_subprocess)
    assert is_correct(_norm_run("def f(x):\n    return 1 + x\n"), expected) is True


def test_incorrect_when_diverges():
    expected = expected_outputs(PROBLEM, run_batch_in_subprocess)
    assert is_correct(_norm_run("def f(x):\n    return x\n"), expected) is False


def test_has_error_false_when_all_inputs_return_cleanly():
    assert has_error(_norm_run("def f(x):\n    return x + 1\n")) is False


def test_has_error_true_when_any_input_raises():
    # raises on x == 2 (ZeroDivisionError), returns fine otherwise -> any-input error
    code = "def f(x):\n    return x // (x - 2)\n"
    assert has_error(_norm_run(code)) is True


def test_has_error_true_when_code_does_not_load():
    # undefined name -> every input errors at call time
    assert has_error(_norm_run("def f(x):\n    return undefined_name\n")) is True


def test_count_outcomes_all_pass():
    expected = expected_outputs(PROBLEM, run_batch_in_subprocess)
    assert count_outcomes(_norm_run("def f(x):\n    return 1 + x\n"), expected) == \
        {"total": 3, "pass": 3, "fail": 0, "error": 0}


def test_count_outcomes_wrong_value_is_fail():
    expected = expected_outputs(PROBLEM, run_batch_in_subprocess)
    # correct except on x == 2 (returns a wrong value, not an error) -> one fail
    code = "def f(x):\n    return x + 1 if x != 2 else 0\n"
    assert count_outcomes(_norm_run(code), expected) == \
        {"total": 3, "pass": 2, "fail": 1, "error": 0}


def test_count_outcomes_raise_is_error():
    expected = expected_outputs(PROBLEM, run_batch_in_subprocess)
    # raises on x == 2 (ZeroDivisionError) -> one error, the rest pass
    code = "def f(x):\n    return x + 1 if x != 2 else 1 // 0\n"
    assert count_outcomes(_norm_run(code), expected) == \
        {"total": 3, "pass": 2, "fail": 0, "error": 1}


def test_count_outcomes_all_error_when_code_does_not_load():
    expected = expected_outputs(PROBLEM, run_batch_in_subprocess)
    assert count_outcomes(_norm_run("def f(x):\n    return undefined_name\n"), expected) == \
        {"total": 3, "pass": 0, "fail": 0, "error": 3}
