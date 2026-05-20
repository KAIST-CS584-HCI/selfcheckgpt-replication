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


def load_dataset(path: str | os.PathLike[str]) -> list[PassageInstance]:
    with open(path) as f:
        return [PassageInstance.from_dict(item) for item in json.load(f)]


def save_result(result: PassageResult, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as tmp:
        json.dump(result.to_dict(), tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


class ScoreRunner:
    def __init__(self, scorer: BaseScorer) -> None:
        self.scorer = scorer

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

        dataset = load_dataset(dataset_path or self.scorer.default_dataset_path)
        out_dir = Path(output_dir) if output_dir is not None else self.default_output_dir()

        written: list[int] = []
        for idx in tqdm(indices, desc=f"{self.scorer.method_name} scoring"):
            output_path = out_dir / f"{idx}.json"
            if output_path.exists() and not overwrite:
                print(f"  Skipping {idx} (already exists)")
                continue

            result = self.scorer.score(idx, dataset[idx])
            save_result(result, output_path)
            written.append(idx)

        print(f"\nDone: {len(written)} entries.")
        return written
