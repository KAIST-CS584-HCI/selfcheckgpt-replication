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
