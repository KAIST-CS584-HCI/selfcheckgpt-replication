from __future__ import annotations
import multiprocessing as mp
import queue as _queue
import time


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
        p.kill()  # SIGKILL: untrusted code may ignore SIGTERM (terminate)
        p.join()
        return ("timeout", None)
    try:
        return q.get_nowait()
    except Exception:
        return ("err", None)


def _batch_worker(code: str, entry_point: str, args_list: list, timeout: float, q) -> None:
    import signal

    def _on_alarm(signum, frame):
        raise TimeoutError()

    try:
        ns: dict = {}
        exec(code, ns)
        fn = ns[entry_point]
    except Exception as exc:  # noqa: BLE001 — module/def failure fails every input
        for _ in args_list:
            q.put(("err", repr(exc)))
        return

    signal.signal(signal.SIGALRM, _on_alarm)
    for args in args_list:
        try:
            signal.setitimer(signal.ITIMER_REAL, timeout)  # per-input wall-clock budget
            try:
                outcome = ("ok", fn(*args))
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        except TimeoutError:
            outcome = ("timeout", None)
        except Exception as exc:  # noqa: BLE001 — any failure of untrusted code is an "err"
            outcome = ("err", repr(exc))
        q.put(outcome)


def run_batch_in_subprocess(code: str, entry_point: str, args_list: list, timeout: float = 5.0):
    """Run entry_point(*args) for every args in `args_list` in ONE spawned process.

    One spawn per implementation instead of one per input. Each call gets its own
    `timeout` budget via an in-process SIGALRM, so a single hanging input is marked
    ("timeout", None) without blocking the rest. A hard, uninterruptible hang trips
    the overall deadline: the process is killed and any not-yet-reported inputs
    become ("timeout", None). Returns a list of outcomes aligned to `args_list`,
    each ("ok", value) | ("err", repr) | ("timeout", None).
    """
    args_list = [list(a) for a in args_list]
    n = len(args_list)
    if n == 0:
        return []
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_batch_worker, args=(code, entry_point, args_list, timeout, q))
    p.start()
    outcomes: list = []
    deadline = time.monotonic() + timeout * n + 5.0  # generous overall ceiling
    while len(outcomes) < n:
        try:
            outcomes.append(q.get(timeout=0.2))
        except _queue.Empty:
            if not p.is_alive() or time.monotonic() > deadline:
                break
    if p.is_alive():
        p.kill()
        p.join()
    outcomes.extend(("timeout", None) for _ in range(n - len(outcomes)))
    return outcomes


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
