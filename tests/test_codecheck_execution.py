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
    assert a == b
    assert normalize_output(("err", None)) == normalize_output(("err", None))
    assert normalize_output(("ok", 2)) != normalize_output(("timeout", None))
