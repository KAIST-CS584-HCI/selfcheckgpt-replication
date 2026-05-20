from __future__ import annotations

from dataclasses import dataclass

from replication.evaluation.constants import PAPER_BASELINES, PAPER_LABELS, VARIANT_LABELS
from replication.evaluation.dataset import EvaluationDataset
from replication.evaluation.metrics import VariantMetrics


@dataclass
class ResponseDistribution:
    total: int
    counts: dict[str, int]

    @classmethod
    def from_dataset(cls, dataset: EvaluationDataset) -> "ResponseDistribution":
        counts: dict[str, int] = {}
        total = 0
        for result in dataset.results:
            for sent_responses in result.responses.prompt or []:
                for msg in sent_responses:
                    total += 1
                    key = cls._normalize_response(msg)
                    counts[key] = counts.get(key, 0) + 1
        return cls(total=total, counts=counts)

    @staticmethod
    def _normalize_response(msg: str | None) -> str:
        if msg is None:
            return "null"

        text = msg.lower().strip()
        if text[:3] == "yes":
            return "Yes"
        if text[:2] == "no":
            return "No"
        return msg.strip() if msg.strip() else "(empty)"

    def to_dict(self) -> dict:
        return {"total": self.total, "counts": self.counts}

    @classmethod
    def from_dict(cls, data: dict) -> "ResponseDistribution":
        return cls(total=data["total"], counts=data["counts"])


@dataclass
class EvaluationReport:
    metrics: list[VariantMetrics | dict]
    variants: list[str]
    response_distribution: ResponseDistribution | dict | None = None

    def metric_dicts(self) -> list[dict]:
        return [
            metric.to_dict() if isinstance(metric, VariantMetrics) else metric
            for metric in self.metrics
        ]

    def response_distribution_dict(self) -> dict | None:
        if self.response_distribution is None:
            return None
        if isinstance(self.response_distribution, ResponseDistribution):
            return self.response_distribution.to_dict()
        return self.response_distribution


class EvaluationReportRenderer:
    def render(self, report: EvaluationReport) -> str:
        if not report.variants:
            return "No score data found for the requested variant(s).\n"

        lines: list[str] = []
        response_distribution = report.response_distribution_dict()
        if response_distribution is not None:
            lines.extend(self.render_response_distribution(response_distribution))

        lines.extend(self.render_summary(report.metric_dicts(), report.variants))
        return "\n".join(lines) + "\n"

    def render_response_distribution(self, dist: dict) -> list[str]:
        total = dist["total"]
        counts = dist["counts"]
        if total == 0:
            return []

        yes = counts.get("Yes", 0)
        no = counts.get("No", 0)
        yes_no = yes + no
        other_counts = {key: value for key, value in counts.items() if key not in ("Yes", "No")}

        lines = [
            f"Response distribution (N={total:,} total API calls):",
            f"  Yes   : {yes:>6,}  ({100 * yes / total:5.1f}%)",
            f"  No    : {no:>6,}  ({100 * no / total:5.1f}%)",
            f"  Yes+No: {yes_no:>6,}  ({100 * yes_no / total:5.1f}%)",
        ]
        if other_counts:
            lines.append("  Other :")
            for key, value in sorted(other_counts.items(), key=lambda item: -item[1]):
                lines.append(f"    {key!r:<20} {value:>6,}  ({100 * value / total:5.1f}%)")
        lines.append("")
        return lines

    def render_summary(self, metrics_list: list[dict], variants: list[str]) -> list[str]:
        header = f"{'Method':<28}  {'NonFact':>8}  {'NonFact*':>8}  {'Factual':>8}  {'Pearson':>8}  {'Spearman':>8}"
        sep = "-" * len(header)
        lines: list[str] = []

        if metrics_list:
            first = metrics_list[0]
            lines.extend([
                f"Passages: {first['n_passages']} | Sentences: {first['n_sentences']}",
                "",
            ])

        lines.extend([header, sep])
        for metric in metrics_list:
            label = VARIANT_LABELS[metric["variant"]]
            lines.append(
                f"{label:<28}  "
                f"{metric['nonfact']:>8.2f}  "
                f"{metric['nonfact_star']:>8.2f}  "
                f"{metric['factual']:>8.2f}  "
                f"{metric['pearson']:>8.2f}  "
                f"{metric['spearman']:>8.2f}"
            )

        lines.extend([sep, "Paper (Table 2, N=20):"])
        for variant in variants:
            baseline = PAPER_BASELINES[variant]
            label = PAPER_LABELS[variant]
            lines.append(
                f"{label:<28}  "
                f"{baseline['nonfact']:>8.2f}  "
                f"{baseline['nonfact_star']:>8.2f}  "
                f"{baseline['factual']:>8.2f}  "
                f"{baseline['pearson']:>8.2f}  "
                f"{baseline['spearman']:>8.2f}"
            )

        return lines
