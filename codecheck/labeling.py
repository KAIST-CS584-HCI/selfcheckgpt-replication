from __future__ import annotations
from codecheck.execution import normalize_output


def expected_outputs(problem, harness, timeout: float = 5.0) -> list:
    """Normalized outputs of the canonical solution over the problem's inputs."""
    return [
        normalize_output(harness(problem.canonical_solution, problem.entry_point, args, timeout), problem.atol)
        for args in problem.inputs
    ]


def is_correct(main_outputs: list, expected: list) -> bool:
    return len(main_outputs) == len(expected) and all(a == b for a, b in zip(main_outputs, expected))
