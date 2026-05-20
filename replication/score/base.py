from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from tqdm import tqdm

from replication.entity import PassageInstance, PassageResult


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "output"


class BaseScorer(ABC):
    method_name: str
    default_dataset_path: Path

    @abstractmethod
    def score(self, dataset_idx: int, passage: PassageInstance) -> PassageResult:
        raise NotImplementedError


class ScoreIO:
    def __init__(
        self,
        dataset_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)

    def load_dataset(self) -> list[PassageInstance]:
        with open(self.dataset_path) as f:
            return [PassageInstance.from_dict(item) for item in json.load(f)]

    def output_exists(self) -> bool:
        return self.output_path.exists()

    def should_skip(self) -> bool:
        return self.output_exists()

    def save_results(self, results: list[PassageResult]) -> None:
        path = self.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as tmp:
            json.dump([result.to_dict() for result in results], tmp, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, path)


class ScoreRunner:
    def __init__(self, scorer: BaseScorer, score_io: ScoreIO) -> None:
        self.scorer = scorer
        self.score_io = score_io

    def run(self, start: int, end: int) -> list[PassageResult]:
        indices = list(range(start, end))
        if not indices:
            print("Nothing to do.")
            return []

        if self.score_io.should_skip():
            print(f"Skipping {self.score_io.output_path} (already exists)")
            return []

        dataset = self.score_io.load_dataset()

        results: list[PassageResult] = []
        for idx in tqdm(indices, desc=f"{self.scorer.method_name} scoring"):
            result = self.scorer.score(idx, dataset[idx])
            results.append(result)

        self.score_io.save_results(results)
        print(f"\nDone: {len(results)} entries.")
        return results
