from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from replication.entity import PassageResult
from replication.evaluation.constants import LABEL_SCORE


@dataclass
class FlattenedScores:
    scores: np.ndarray
    labels: np.ndarray
    passage_mean: np.ndarray


class EvaluationDataset:
    def __init__(self, results: list[PassageResult]) -> None:
        self.results = results

    @classmethod
    def from_json(cls, path: str | Path) -> "EvaluationDataset":
        with open(path) as f:
            return cls([PassageResult.from_dict(item) for item in json.load(f)])

    def slice(self, start: int | None = None, end: int | None = None) -> "EvaluationDataset":
        start_idx = start if start is not None else 0
        end_idx = end if end is not None else len(self.results) - 1
        return EvaluationDataset(self.results[start_idx:end_idx + 1])

    def scores_for_variant(self, result: PassageResult, variant: str) -> list[float] | None:
        return result.scores.get(variant)

    def active_variants(self, requested: list[str]) -> list[str]:
        return [
            variant
            for variant in requested
            if any(self.scores_for_variant(result, variant) is not None for result in self.results)
        ]

    def results_with_scores(self, variant: str) -> list[PassageResult]:
        return [result for result in self.results if self.scores_for_variant(result, variant) is not None]

    def flatten(self, variant: str) -> FlattenedScores:
        scores, labels, passage_mean = [], [], []
        for result in self.results_with_scores(variant):
            sent_scores = self.scores_for_variant(result, variant)
            gold = np.array([LABEL_SCORE[annotation] for annotation in result.annotation])
            mean_label = float(gold.mean())
            scores.extend(sent_scores or [])
            labels.extend(gold.tolist())
            passage_mean.extend([mean_label] * len(sent_scores or []))

        return FlattenedScores(
            scores=np.asarray(scores),
            labels=np.asarray(labels),
            passage_mean=np.asarray(passage_mean),
        )
