from __future__ import annotations
from codecheck.execution import normalize_output


def expected_outputs(problem, harness, timeout: float = 5.0) -> list:
    """Normalized outputs of the canonical solution over the problem's inputs.

    `harness` is a batch harness: one call runs all inputs and returns a list of
    outcomes aligned to `problem.inputs`.
    """
    outcomes = harness(problem.canonical_solution, problem.entry_point, problem.inputs, timeout)
    return [normalize_output(o, problem.atol) for o in outcomes]


def is_correct(main_outputs: list, expected: list) -> bool:
    return len(main_outputs) == len(expected) and all(a == b for a, b in zip(main_outputs, expected))


def count_outcomes(main_outputs: list, expected: list) -> dict[str, int]:
    """Per-input breakdown of the main vs the canonical, partitioned to sum to `total`:
    `pass` (matched the canonical), `error` (a non-match where the main raised/timed out),
    `fail` (a non-match where the main returned a wrong value). `pass` uses the same
    comparison as `is_correct`."""
    counts = {"total": len(expected), "pass": 0, "fail": 0, "error": 0}
    for main, exp in zip(main_outputs, expected):
        if main == exp:
            counts["pass"] += 1
        elif main[0] == "status":   # ("status", "err"|"timeout") = raised or timed out
            counts["error"] += 1
        else:
            counts["fail"] += 1
    return counts


def has_error(main_outputs: list) -> bool:
    """True if the implementation raised or timed out on ANY input.

    `normalize_output` maps a clean return to `("value", ...)` and a failure (exception
    or timeout) to `("status", ...)`, so a non-ok status anywhere means the code did not
    run cleanly. Distinguishes an error-out from a ran-but-wrong (semantic) result."""
    return any(outcome[0] == "status" for outcome in main_outputs)
