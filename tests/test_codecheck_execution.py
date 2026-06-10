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


# String hashing is seed-dependent, so `set` iteration order differs across separately
# spawned interpreters unless the hash seed is pinned (execution.py pins PYTHONHASHSEED).
# Without the pin, identical code returns differently-ordered tuples per process, which
# inflates the consistency score and mislabels correct code. Two spawns must now agree.
SET_ORDER = "def f(xs):\n    return tuple(set(xs))\n"


def test_batch_set_ordering_is_deterministic_across_spawns():
    inp = [["banana", "apple", "cherry", "date", "fig", "grape", "kiwi", "lemon"]]
    a = run_batch_in_subprocess(SET_ORDER, "f", inp, timeout=5.0)
    b = run_batch_in_subprocess(SET_ORDER, "f", inp, timeout=5.0)
    assert a == b


def test_canonical_mixed_type_set_is_order_independent():
    # untrusted code may return a set of mixed types; sorted() would raise str<int.
    a = normalize_output(("ok", {1, "x", 2}), atol=1e-6)
    b = normalize_output(("ok", {"x", 2, 1}), atol=1e-6)  # same set, different literal order
    assert a == b


def test_canonical_mixed_type_dict_keys():
    a = normalize_output(("ok", {1: "a", "k": 2}), atol=0)
    b = normalize_output(("ok", {"k": 2, 1: "a"}), atol=0)
    assert a == b


# A sample the in-worker SIGALRM cannot stop: it blocks SIGALRM delivery, so setitimer
# fires but the handler never runs and the input emits no result at all. This is the real
# freeze (a worker pegged at 100% CPU with no new output). The parent must enforce its own
# wall-clock bound and SIGKILL instead of waiting the timeout*n ceiling.
BLOCK_ALARM = (
    "def f(x):\n"
    "    import signal\n"
    "    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})\n"
    "    while True:\n"
    "        pass\n"
)


def test_batch_kills_uninterruptible_hang_without_waiting_for_full_ceiling():
    import time
    n = 50
    timeout = 0.3
    args = [[i] for i in range(n)]  # the first input wedges the worker, emitting nothing
    start = time.monotonic()
    out = run_batch_in_subprocess(BLOCK_ALARM, "f", args, timeout=timeout)
    elapsed = time.monotonic() - start
    # Old behaviour waited the timeout*n (=15s) ceiling; a parent-side idle timeout
    # must kill far sooner. Generous bound to stay robust under load.
    assert elapsed < timeout * n * 0.5
    assert len(out) == n
    assert all(o == ("timeout", None) for o in out)


def test_batch_raises_on_worker_crash_before_output():
    # os._exit(1) during module exec: the worker dies before any q.put and exits non-zero,
    # which is an infra/bootstrap failure, not a code error -> must surface, not silently
    # return all-timeout.
    import pytest
    crash = "import os\nos._exit(1)\n"
    with pytest.raises(RuntimeError):
        run_batch_in_subprocess(crash, "f", [[1], [2]], timeout=2.0)
