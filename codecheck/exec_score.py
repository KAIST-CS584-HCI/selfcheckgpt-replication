from __future__ import annotations


def exec_inconsistency(main_outputs: list, sample_outputs: list[list]) -> float:
    """1 - mean per-sample agreement with main, over shared inputs. Range [0, 1]."""
    if not sample_outputs:
        return 0.0
    n_inputs = len(main_outputs)
    if n_inputs == 0:
        return 0.0
    agreements = []
    for sample in sample_outputs:
        matches = sum(1 for a, b in zip(main_outputs, sample) if a == b)
        agreements.append(matches / n_inputs)
    return 1.0 - sum(agreements) / len(agreements)
