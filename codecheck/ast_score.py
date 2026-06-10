from __future__ import annotations
import ast
from collections import Counter

import zss

# Relabel/insert/delete each cost 1 (a label either matches or it doesn't). Passed to
# zss.simple_distance, this also makes insert/delete unit-cost (label_dist('', x) == 1),
# replacing zss's default *string*-edit distance between labels — so the raw distance is
# bounded by size(a) + size(b) and the normalized score stays in [0, 1].
def _unit_label_dist(a: str, b: str) -> int:
    return 0 if a == b else 1


def ast_fingerprint(code: str) -> Counter | None:
    """Multiset of AST node-type names for `code`, or None if it does not parse.

    Only the node *type* is counted (e.g. `Name`, `BinOp`, `Constant`), never the
    identifier text or literal value — so renaming variables or changing constants
    leaves the fingerprint unchanged. This captures structure, normalized for
    identifier renaming and literal differences (the methods-doc requirement).
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        # SyntaxError: malformed code. ValueError: e.g. source with null bytes.
        # TypeError: a non-str input (e.g. None from a failed extraction).
        return None
    if not tree.body:
        # Empty, whitespace-only, or comment-only source parses to a body-less
        # module: a non-answer, not a structure to compare. Treat as unparseable.
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


def ast_to_tree(code: str) -> "zss.Node | None":
    """Convert `code` to a zss tree labeled by AST node-type name, or None if it does
    not parse / is body-less.

    Labels are node-type names only (e.g. `BinOp`, `Call`) — never identifier text or
    literal values — so the tree is invariant to variable renaming and literal changes,
    matching `ast_fingerprint`. Parse-failure handling mirrors it exactly.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return None
    if not tree.body:
        return None
    return _to_zss(tree)


def _to_zss(node: ast.AST) -> "zss.Node":
    z = zss.Node(type(node).__name__)
    for child in ast.iter_child_nodes(node):
        z.addkid(_to_zss(child))
    return z


def _tree_size(node: "zss.Node") -> int:
    return 1 + sum(_tree_size(child) for child in node.children)


def ted_dissimilarity(main_tree: "zss.Node", sample_tree: "zss.Node") -> float:
    """Zhang-Shasha tree edit distance between two label trees, normalized to [0,1].

    Unlike the bag-of-node-types Jaccard, this compares actual tree shape, so two
    programs with the same node-type counts but different nesting score apart. 0.0 =
    identical structure; higher = more divergent (same scale/direction as Jaccard,
    Exec, Prompt). Normalized as distance / (size_main + size_sample), which is ≤ 1
    because deleting all of one tree and inserting all of the other costs exactly that.
    """
    total = _tree_size(main_tree) + _tree_size(sample_tree)
    if total == 0:
        return 0.0
    distance = zss.simple_distance(main_tree, sample_tree, label_dist=_unit_label_dist)
    return distance / total


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
