from __future__ import annotations
from dataclasses import dataclass


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
    exec_score: float
    is_correct: bool
    main_code: str
    sample_codes: list[str]
    n_inputs: int = 0          # size of the shared input set all impls ran on

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "exec_score": self.exec_score, "is_correct": self.is_correct,
            "main_code": self.main_code, "sample_codes": self.sample_codes, "n_inputs": self.n_inputs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CodeResult":
        return cls(
            task_id=d["task_id"], exec_score=d["exec_score"], is_correct=d["is_correct"],
            main_code=d["main_code"], sample_codes=d["sample_codes"], n_inputs=d.get("n_inputs", 0),
        )
