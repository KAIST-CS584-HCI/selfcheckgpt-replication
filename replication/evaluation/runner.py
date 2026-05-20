from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from replication.evaluation.constants import VARIANTS
from replication.evaluation.dataset import EvaluationDataset
from replication.evaluation.metrics import MetricSuite
from replication.evaluation.report import EvaluationReport, ResponseDistribution


@dataclass
class EvaluationConfig:
    results_path: str | Path
    variant: str = "all"
    start: int | None = None
    end: int | None = None


class EvaluationRunner:
    def __init__(self, metric_suite: MetricSuite | None = None) -> None:
        self.metric_suite = metric_suite or MetricSuite()

    def run(self, config: EvaluationConfig) -> EvaluationReport:
        dataset = EvaluationDataset.from_json(config.results_path).slice(
            start=config.start,
            end=config.end,
        )
        requested = VARIANTS if config.variant == "all" else [config.variant]
        variants = dataset.active_variants(requested)

        if not variants:
            return EvaluationReport(metrics=[], variants=[])

        response_distribution = None
        if "prompt" in variants:
            response_distribution = ResponseDistribution.from_dataset(dataset)

        metrics = [
            self.metric_suite.evaluate_variant(dataset, variant)
            for variant in variants
        ]
        return EvaluationReport(
            metrics=metrics,
            variants=variants,
            response_distribution=response_distribution,
        )
