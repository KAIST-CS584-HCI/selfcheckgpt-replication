from __future__ import annotations
from dataclasses import dataclass, field


def _empty_count() -> dict[str, int]:
    return {"total": 0, "pass": 0, "fail": 0, "error": 0}


@dataclass
class CodeProblem:
    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str
    inputs: list[list]
    atol: float = 1e-6

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "prompt": self.prompt, "entry_point": self.entry_point,
            "canonical_solution": self.canonical_solution, "inputs": self.inputs, "atol": self.atol,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CodeProblem":
        return cls(
            task_id=d["task_id"], prompt=d["prompt"], entry_point=d["entry_point"],
            canonical_solution=d["canonical_solution"], inputs=d["inputs"], atol=d.get("atol", 1e-6),
        )


@dataclass
class CodeResult:
    task_id: str
    scores: dict[str, float]      # method name -> inconsistency score in [0, 1]
    is_correct: bool
    main_code: str
    sample_codes: list[str]
    prompt_responses: list[str] | None = None   # raw judge text per sample (prompt method only)
    prompt: str = ""              # the original problem prompt shown to the model
    is_error: bool = False        # main raised/timed out on any input (vs ran-but-wrong)
    # per-input breakdown vs the canonical: {total, pass (matched), fail (wrong value),
    # error (raised/timed out)}; pass + fail + error == total
    count: dict[str, int] = field(default_factory=_empty_count)

    @property
    def exec_score(self) -> float:
        return self.scores.get("exec", float("nan"))

    def to_dict(self) -> dict:
        d = {
            "task_id": self.task_id, "scores": self.scores, "is_correct": self.is_correct,
            "is_error": self.is_error, "count": self.count, "prompt": self.prompt,
            "main_code": self.main_code, "sample_codes": self.sample_codes,
        }
        if self.prompt_responses is not None:
            d["prompt_responses"] = self.prompt_responses
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CodeResult":
        if "scores" in d:
            scores = dict(d["scores"])
        elif "exec_score" in d:               # iteration-1 artifact back-compat
            scores = {"exec": d["exec_score"]}
        else:
            scores = {}
        return cls(
            task_id=d["task_id"], scores=scores, is_correct=d["is_correct"],
            main_code=d["main_code"], sample_codes=d["sample_codes"],
            prompt_responses=d.get("prompt_responses"),
            prompt=d.get("prompt", ""), is_error=d.get("is_error", False),
            count=d.get("count") or _empty_count(),
        )
