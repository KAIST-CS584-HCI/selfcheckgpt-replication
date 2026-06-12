from __future__ import annotations
import logging
import os
import subprocess
import sys
import tempfile

logger = logging.getLogger("codecheck.execution")

# Pin the hash seed so stdin->stdout programs that iterate sets/dicts print in a stable order
# across separately spawned interpreters (same rationale as the function-call sandbox).
_CHILD_ENV = {**os.environ, "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "0")}


def normalize_stdout(s: str) -> str:
    """Canonicalize program stdout for comparison: unify line endings, strip trailing
    whitespace on each line, and drop leading/trailing blank lines. Keeps a behavioral
    diff meaningful without tripping on cosmetic whitespace. Numeric/formatting-sensitive
    output is still compared as exact strings (see the iteration plan's open note)."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in s.split("\n")).strip("\n")


def _err_repr(stderr: str, returncode: int) -> str:
    """A short error tag for a non-zero exit: the last stderr line (usually the exception)
    or the exit code when stderr is empty."""
    lines = [ln for ln in stderr.splitlines() if ln.strip()]
    return lines[-1] if lines else f"exit {returncode}"


def _run_one(prog: str, stdin: str, timeout: float) -> tuple:
    try:
        proc = subprocess.run(
            [sys.executable, prog], input=stdin, capture_output=True, text=True,
            timeout=timeout, env=_CHILD_ENV,
        )
    except subprocess.TimeoutExpired:
        return ("timeout", None)
    if proc.returncode != 0:
        return ("err", _err_repr(proc.stderr, proc.returncode))
    return ("ok", normalize_stdout(proc.stdout))


def run_stdio_batch_in_subprocess(code: str, _entry_point: str, stdin_list: list,
                                  timeout: float = 5.0) -> list:
    """Whole-program stdin->stdout harness, mirroring the batch-harness signature
    (`run_batch_in_subprocess`). `_entry_point` is ignored (stdio programs have no callable).
    Each stdin runs the program in a fresh `python` subprocess with its own `timeout` budget;
    returns a list of outcomes aligned to `stdin_list`, each
    ("ok", normalized_stdout) | ("err", repr) | ("timeout", None)."""
    if not stdin_list:
        return []
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        prog = f.name
    try:
        outcomes = [_run_one(prog, str(stdin), timeout) for stdin in stdin_list]
    finally:
        os.unlink(prog)
    n_timeouts = sum(1 for o in outcomes if o[0] == "timeout")
    if n_timeouts:
        logger.warning("stdio program: %d/%d inputs timed out (>%gs each)",
                       n_timeouts, len(stdin_list), timeout)
    return outcomes
