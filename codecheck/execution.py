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
        p.kill()  # SIGKILL: untrusted code may ignore SIGTERM (terminate)
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
