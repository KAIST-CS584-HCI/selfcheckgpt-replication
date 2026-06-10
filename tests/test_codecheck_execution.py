from codecheck.execution import run_in_subprocess, run_batch_in_subprocess, normalize_output

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
    assert a == b
    assert normalize_output(("err", None)) == normalize_output(("err", None))
    assert normalize_output(("ok", 2)) != normalize_output(("timeout", None))


HANG_NEG = "def f(x):\n    while x < 0:\n        pass\n    return x\n"


def test_batch_runs_all_inputs_in_order():
    out = run_batch_in_subprocess(ADD, "f", [[1], [2], [3]], timeout=5.0)
    assert out == [("ok", 2), ("ok", 3), ("ok", 4)]


def test_batch_empty_inputs():
    assert run_batch_in_subprocess(ADD, "f", [], timeout=5.0) == []


def test_batch_per_input_error_and_value():
    out = run_batch_in_subprocess(BOOM if False else ADD, "f", [[1]], timeout=5.0)
    assert out == [("ok", 2)]
    err = run_batch_in_subprocess(BOOM, "f", [[1], [2]], timeout=5.0)
    assert [o[0] for o in err] == ["err", "err"]


def test_batch_timeout_isolates_per_input():
    # first input hangs -> timeout; later inputs still run (in-process SIGALRM)
    out = run_batch_in_subprocess(HANG_NEG, "f", [[-1], [7]], timeout=1.0)
    assert out[0][0] == "timeout"
    assert out[1] == ("ok", 7)


def test_batch_module_load_failure_marks_all():
    out = run_batch_in_subprocess("def broken(:\n", "f", [[1], [2]], timeout=2.0)
    assert [o[0] for o in out] == ["err", "err"]
