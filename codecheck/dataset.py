from __future__ import annotations
import json
import random
from pathlib import Path

from evalplus.data import get_human_eval_plus, get_mbpp_plus

from codecheck.models import CodeProblem

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "data" / "mbpp_plus.json"
HUMAN_EVAL_CACHE = REPO_ROOT / "data" / "human_eval_plus.json"
CODEHALU_CACHE = REPO_ROOT / "data" / "codehalu.json"
CODEHALU_DATASET = "Yuchen111/CodeHaluEval"


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


# CodeHaluEval ships as 8 JSON-list files, one per hallucination category. `load_dataset`
# fails to merge them (a column inferred as null in one file conflicts with string in
# another), so download and parse the raw JSON directly.
CODEHALU_FILES = (
    "calculate_boundary_hallucination.json", "data_compliance_hallucination.json",
    "external_source_hallucination.json", "identification_hallucination.json",
    "logic_breakdown.json", "logic_deviation.json",
    "physical_constraint_hallucination.json", "structural_access_hallucination.json",
)


def _load_codehalu_rows() -> list[dict]:
    """Download and concatenate the CodeHaluEval rows from HuggingFace (one row per test
    case, across all 8 category files). Isolated so tests substitute a tiny in-memory fake."""
    from huggingface_hub import hf_hub_download
    rows: list[dict] = []
    for fname in CODEHALU_FILES:
        path = hf_hub_download(CODEHALU_DATASET, fname, repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            rows.extend(json.load(f))
    return rows


def _is_stdio(row: dict) -> bool:
    """True for whole-program stdin->stdout tasks (the ones we run); a non-empty `fn_name`
    marks the LeetCode-style call tasks, which the stdio harness cannot drive."""
    fn = row.get("fn_name")
    return fn is None or (isinstance(fn, str) and fn.strip().lower() in ("", "null", "none"))


def _stdio_text(raw) -> str:
    """Coerce a CodeHaluEval `input`/`output` field to a stdin/stdout string. Most rows store
    a raw string; a few store the lines as a list, which join with newlines."""
    if isinstance(raw, list):
        return "\n".join(str(line) for line in raw)
    return str(raw)


def _first_solution(raw) -> str:
    """First reference program from a row's `solutions` (a JSON-encoded list string; `''`
    when none ships). Reference only — the canonical never runs (expected is pre-supplied)."""
    if not raw:
        return ""
    solutions = json.loads(raw) if isinstance(raw, str) else raw
    return solutions[0] if solutions else ""


def _codehalu_problem(task_id, rows: list[dict], max_cases: int) -> CodeProblem:
    """Build one stdin->stdout CodeProblem from a task's test-case rows. `input`/`output` are
    raw stdin/stdout strings; expected outputs ship with the data, so `expected` is pre-filled
    (normalized) and the canonical is reference-only."""
    from codecheck.execution.stdio_sandbox import normalize_stdout
    first = rows[0]
    rows = rows[:max_cases]   # bound runtime: a task can carry hundreds of cases
    return CodeProblem(
        task_id=f"CodeHalu/{task_id}",
        prompt=first["question"],
        entry_point="",                                   # stdio programs have no callable
        canonical_solution=_first_solution(first["solutions"]),
        inputs=[_stdio_text(r["input"]) for r in rows],   # raw stdin (str, or lines list)
        atol=0.0,                                         # stdout compared as exact strings
        expected=[("value", normalize_stdout(_stdio_text(r["output"]))) for r in rows],
    )


def load_codehalu_eval(
    limit: int | None = None,
    randomize: bool = False,
    seed: int | None = None,
    index: int | None = None,
    max_cases: int = 25,
    cache_path: Path = CODEHALU_CACHE,
) -> list[CodeProblem]:
    """Load CodeHaluEval stdin->stdout tasks. Rows arrive one per test case; group by
    `task_id`, keep the stdio tasks (`fn_name` empty), and cap each task at `max_cases` test
    cases to bound runtime. Expected outputs ship with the data (no canonical run).
    Selection/index semantics match `load_mbpp_plus`."""
    grouped: dict = {}
    for row in _load_codehalu_rows():
        if _is_stdio(row):
            grouped.setdefault(row["task_id"], []).append(row)
    problems = [_codehalu_problem(tid, rows, max_cases) for tid, rows in grouped.items()]
    return _finalize(problems, limit, randomize, seed, index, cache_path)
