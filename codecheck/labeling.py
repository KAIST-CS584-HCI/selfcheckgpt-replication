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
