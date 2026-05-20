from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from tqdm import tqdm

from replication.entity import PassageInstance, PassageResult


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results"


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
        output_dir: str | os.PathLike[str],
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)

    def load_dataset(self) -> list[PassageInstance]:
        with open(self.dataset_path) as f:
            return [PassageInstance.from_dict(item) for item in json.load(f)]

    def result_path(self, index: int) -> Path:
        return self.output_dir / f"{index}.json"

    def result_exists(self, index: int) -> bool:
        return self.result_path(index).exists()

    def save_result(self, index: int, result: PassageResult) -> None:
        path = self.result_path(index)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as tmp:
            json.dump(result.to_dict(), tmp, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, path)


class ScoreRunner:
    def __init__(self, scorer: BaseScorer, score_io: ScoreIO | None = None) -> None:
        self.scorer = scorer
        self.score_io = score_io

    def default_output_dir(self) -> Path:
        return DEFAULT_RESULTS_ROOT / self.scorer.method_name

    def run(
        self,
        start: int,
        end: int,
        dataset_path: str | os.PathLike[str] | None = None,
        output_dir: str | os.PathLike[str] | None = None,
        overwrite: bool = False,
    ) -> list[int]:
        indices = list(range(start, end))
        if not indices:
            print("Nothing to do.")
            return []

        score_io = self.score_io or ScoreIO(
            dataset_path=dataset_path or self.scorer.default_dataset_path,
            output_dir=output_dir or self.default_output_dir(),
        )
        dataset = score_io.load_dataset()

        written: list[int] = []
        for idx in tqdm(indices, desc=f"{self.scorer.method_name} scoring"):
            if score_io.result_exists(idx) and not overwrite:
                print(f"  Skipping {idx} (already exists)")
                continue

            result = self.scorer.score(idx, dataset[idx])
            score_io.save_result(idx, result)
            written.append(idx)

        print(f"\nDone: {len(written)} entries.")
        return written
