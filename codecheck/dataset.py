from __future__ import annotations
import json
import random
from pathlib import Path

from evalplus.data import get_mbpp_plus

from codecheck.models import CodeProblem

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "data" / "mbpp_plus.json"


def _to_problem(item: dict) -> CodeProblem:
    return CodeProblem(
        task_id=item["task_id"],
        prompt=item["prompt"],
        entry_point=item["entry_point"],
        canonical_solution=item["canonical_solution"],
        inputs=list(item.get("base_input", [])) + list(item.get("plus_input", [])),
        atol=item.get("atol", 1e-6),
    )


def _select(problems: list[CodeProblem], limit: int, randomize: bool, seed: int | None) -> list[CodeProblem]:
    if not randomize:
        return problems[:limit]
    return random.Random(seed).sample(problems, min(limit, len(problems)))


def load_mbpp_plus(
    limit: int | None = None,
    randomize: bool = False,
    seed: int | None = None,
    index: int | None = None,
    cache_path: Path = DEFAULT_CACHE,
) -> list[CodeProblem]:
    """Load MBPP+ problems. With a limit, take the first `limit` in dataset order by
    default (set randomize=True for a random sample; pass `seed` for a reproducible
    one). With `index` set, return only the single problem at that 0-based dataset
    position (limit/randomize/seed are ignored)."""
    problems = [_to_problem(item) for item in get_mbpp_plus().values()]
    if index is not None:
        if not 0 <= index < len(problems):
            raise IndexError(f"--index {index} out of range (dataset has {len(problems)} problems)")
        problems = [problems[index]]
    elif limit is not None:
        problems = _select(problems, limit, randomize, seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # default=str keeps the write-only debug cache tolerant of non-JSON inputs
    # (e.g. complex numbers in Mbpp/124, Mbpp/252); the cache is never read back.
    cache_path.write_text(
        json.dumps([p.to_dict() for p in problems], indent=2, default=str),
        encoding="utf-8",
    )
    return problems
