from __future__ import annotations
import json
import random
from pathlib import Path

from evalplus.data import get_human_eval_plus, get_mbpp_plus

from codecheck.models import CodeProblem

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "data" / "mbpp_plus.json"
HUMAN_EVAL_CACHE = REPO_ROOT / "data" / "human_eval_plus.json"


def _to_problem(item: dict, *, canonical: str | None = None) -> CodeProblem:
    """Map one EvalPlus item to a CodeProblem. `canonical` overrides the stored
    canonical_solution (HumanEval+ ships the body only, so it must be assembled as
    prompt + body to be runnable)."""
    return CodeProblem(
        task_id=item["task_id"],
        prompt=item["prompt"],
        entry_point=item["entry_point"],
        canonical_solution=canonical if canonical is not None else item["canonical_solution"],
        inputs=list(item.get("base_input", [])) + list(item.get("plus_input", [])),
        atol=item.get("atol", 1e-6),
    )


def _select(problems: list[CodeProblem], limit: int, randomize: bool, seed: int | None) -> list[CodeProblem]:
    if not randomize:
        return problems[:limit]
    return random.Random(seed).sample(problems, min(limit, len(problems)))


def _finalize(problems: list[CodeProblem], limit: int | None, randomize: bool,
              seed: int | None, index: int | None, cache_path: Path) -> list[CodeProblem]:
    """Shared tail for every loader: apply index/limit selection, then write the
    write-only debug cache. `index` selects a single problem (limit/randomize/seed
    ignored); otherwise `limit` takes the first/random `limit`."""
    cache_path = Path(cache_path)   # tolerate a str path from programmatic callers
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
    return _finalize(problems, limit, randomize, seed, index, cache_path)


def load_human_eval_plus(
    limit: int | None = None,
    randomize: bool = False,
    seed: int | None = None,
    index: int | None = None,
    cache_path: Path = HUMAN_EVAL_CACHE,
) -> list[CodeProblem]:
    """Load HumanEval+ problems (same function-call interface as MBPP+). HumanEval+
    stores `canonical_solution` as the body only, with `prompt` holding the
    signature+docstring, so the runnable canonical is assembled as prompt + body.
    Selection/index semantics match `load_mbpp_plus`."""
    problems = [
        _to_problem(item, canonical=item["prompt"] + item["canonical_solution"])
        for item in get_human_eval_plus().values()
    ]
    return _finalize(problems, limit, randomize, seed, index, cache_path)
