from __future__ import annotations
import ast
from collections import Counter


def ast_fingerprint(code: str) -> Counter | None:
    """Multiset of AST node-type names for `code`, or None if it does not parse.

    Only the node *type* is counted (e.g. `Name`, `BinOp`, `Constant`), never the
    identifier text or literal value — so renaming variables or changing constants
    leaves the fingerprint unchanged. This captures structure, normalized for
    identifier renaming and literal differences (the methods-doc requirement).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    return Counter(type(node).__name__ for node in ast.walk(tree))


def ast_dissimilarity(main_fp: Counter, sample_fp: Counter) -> float:
    """1 - multiset Jaccard of two fingerprints. 0.0 = identical structure,
    1.0 = no shared node types. Two empty fingerprints count as identical."""
    intersection = sum((main_fp & sample_fp).values())
    union = sum((main_fp | sample_fp).values())
    if union == 0:
        return 0.0
    return 1.0 - intersection / union


class ASTScorer:
    """Structural-consistency scorer (SelfCheck-AST). evaluate() returns the mean
    structural dissimilarity of the samples vs the main implementation, on the same
    [0,1] scale and direction as Exec/Prompt (higher = more likely incorrect).
    `parse_failures` accumulates implementations that do not parse, mirroring
    PromptJudge so the CLI can report it.
    """

    def __init__(self) -> None:
        self.parse_failures = 0

    def evaluate(self, main_code: str, sample_codes: list[str]) -> tuple[float, list[float]]:
        """(mean_dissimilarity, per_sample_dissimilarities)."""
        if not sample_codes:
            return 0.0, []
        main_fp = ast_fingerprint(main_code)
        if main_fp is None:
            # The T=0 main failed to parse: structure is unverifiable, so treat every
            # sample as maximally divergent and count one parse failure for the main.
            self.parse_failures += 1
            return 1.0, [1.0] * len(sample_codes)
        per_sample: list[float] = []
        for code in sample_codes:
            sample_fp = ast_fingerprint(code)
            if sample_fp is None:
                self.parse_failures += 1
                per_sample.append(1.0)
            else:
                per_sample.append(ast_dissimilarity(main_fp, sample_fp))
        return sum(per_sample) / len(per_sample), per_sample

    def score(self, main_code: str, sample_codes: list[str]) -> float:
        return self.evaluate(main_code, sample_codes)[0]
