from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import auc, precision_recall_curve

from replication.evaluation.constants import LABEL_SCORE
from replication.evaluation.dataset import EvaluationDataset, FlattenedScores


@dataclass
class VariantMetrics:
    variant: str
    n_passages: int
    n_sentences: int
    nonfact: float
    nonfact_star: float
    factual: float
    pearson: float
    spearman: float

    def to_dict(self) -> dict:
        return {
            "variant": self.variant,
            "n_passages": self.n_passages,
            "n_sentences": self.n_sentences,
            "nonfact": self.nonfact,
            "nonfact_star": self.nonfact_star,
            "factual": self.factual,
            "pearson": self.pearson,
            "spearman": self.spearman,
        }


class MetricSuite:
    def auc_pr_nonfact(self, flattened: FlattenedScores) -> float:
        y_true = (flattened.labels > 0).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, flattened.scores)
        return float(auc(recall, precision))

    def auc_pr_nonfact_star(self, flattened: FlattenedScores) -> float:
        mask = flattened.passage_mean < 1.0
        y_true = (flattened.labels[mask] == 1.0).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, flattened.scores[mask])
        return float(auc(recall, precision))

    def auc_pr_factual(self, flattened: FlattenedScores) -> float:
        y_true = (flattened.labels == 0.0).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, -flattened.scores)
        return float(auc(recall, precision))

    def passage_correlations(self, dataset: EvaluationDataset, variant: str) -> tuple[float, float]:
        pred, gold = [], []
        for result in dataset.results_with_scores(variant):
            sent_scores = dataset.scores_for_variant(result, variant)
            pred.append(float(np.mean(sent_scores)))
            gold.append(float(np.mean([LABEL_SCORE[annotation] for annotation in result.annotation])))

        pearson = float(pearsonr(pred, gold).statistic)
        spearman = float(spearmanr(pred, gold).statistic)
        return pearson, spearman

    def evaluate_variant(self, dataset: EvaluationDataset, variant: str) -> VariantMetrics:
        flattened = dataset.flatten(variant)
        pearson, spearman = self.passage_correlations(dataset, variant)
        return VariantMetrics(
            variant=variant,
            n_passages=len(dataset.results_with_scores(variant)),
            n_sentences=len(flattened.scores),
            nonfact=self.auc_pr_nonfact(flattened) * 100,
            nonfact_star=self.auc_pr_nonfact_star(flattened) * 100,
            factual=self.auc_pr_factual(flattened) * 100,
            pearson=pearson * 100,
            spearman=spearman * 100,
        )
