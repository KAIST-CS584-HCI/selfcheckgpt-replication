from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from replication.entity import PassageResult


SLICE_FILENAME_RE = re.compile(r"^(bert|nli|prompt)-(\d+)-to-(\d+)\.json$")
VARIANTS = ("bert", "nli", "prompt")


@dataclass
class AggregateConfig:
    dataset_path: Path
    inputs_dir: Path
    output_path: Path


@dataclass
class VariantCoverage:
    variant: str
    done: int
    total: int
    missing_indices: list[int]

    def missing_ranges(self) -> str:
        return compact_ranges(self.missing_indices)

    def format_line(self) -> str:
        if not self.missing_indices:
            return f"{self.variant}: {self.done}/{self.total} done ✓"
        return f"{self.variant}: {self.done}/{self.total} done, missing {self.missing_ranges()}"


@dataclass
class AggregateReport:
    output_path: Path
    aggregate_size: int
    coverages: list[VariantCoverage] = field(default_factory=list)

    def format(self) -> str:
        lines = [
            f"Aggregated {self.aggregate_size} passages -> {self.output_path}",
        ]
        lines.extend(coverage.format_line() for coverage in self.coverages)
        return "\n".join(lines) + "\n"


def compact_ranges(indices: list[int]) -> str:
    if not indices:
        return ""
    ordered = sorted(set(indices))
    spans: list[str] = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        spans.append(_format_span(start, prev))
        start = prev = value
    spans.append(_format_span(start, prev))
    return ",".join(spans)


def _format_span(start: int, end: int) -> str:
    return f"{start}" if start == end else f"{start}-{end}"


class AggregateRunner:
    def run(self, config: AggregateConfig) -> AggregateReport:
        aggregated = self._load_existing(config.output_path)
        slice_paths = self._discover_slices(config.inputs_dir, config.output_path)
        for slice_path in slice_paths:
            self._merge_slice(aggregated, slice_path)
        self._save(aggregated, config.output_path)

        total = self._dataset_length(config.dataset_path)
        coverages = self._build_coverages(aggregated, total)
        return AggregateReport(
            output_path=config.output_path,
            aggregate_size=len(aggregated),
            coverages=coverages,
        )

    def _load_existing(self, output_path: Path) -> dict[int, PassageResult]:
        if not output_path.exists():
            return {}
        with open(output_path) as f:
            items = json.load(f)
        return {
            item["dataset_idx"]: PassageResult.from_dict(item)
            for item in items
        }

    def _discover_slices(self, inputs_dir: Path, output_path: Path) -> list[Path]:
        if not inputs_dir.exists():
            return []
        slices: list[Path] = []
        output_resolved = output_path.resolve() if output_path.exists() else None
        for entry in inputs_dir.iterdir():
            if not entry.is_file():
                continue
            if output_resolved is not None and entry.resolve() == output_resolved:
                continue
            if not SLICE_FILENAME_RE.match(entry.name):
                continue
            slices.append(entry)
        slices.sort(key=lambda p: p.stat().st_mtime)
        return slices

    def _merge_slice(self, aggregated: dict[int, PassageResult], slice_path: Path) -> None:
        with open(slice_path) as f:
            items = json.load(f)
        for item in items:
            incoming = PassageResult.from_dict(item)
            existing = aggregated.get(incoming.dataset_idx)
            if existing is None:
                aggregated[incoming.dataset_idx] = incoming
                continue
            self._merge_into(existing, incoming, slice_path)

    def _merge_into(
        self,
        existing: PassageResult,
        incoming: PassageResult,
        slice_path: Path,
    ) -> None:
        for variant in VARIANTS:
            new_scores = incoming.scores.get(variant)
            if new_scores is None:
                continue
            old_scores = existing.scores.get(variant)
            if old_scores is not None and old_scores != new_scores:
                print(
                    f"WARN: dataset_idx={existing.dataset_idx} variant={variant} "
                    f"score mismatch, using newer (from {slice_path})"
                )
            setattr(existing.scores, variant, new_scores)

        new_prompt_responses = incoming.responses.get("prompt")
        if new_prompt_responses is not None:
            existing.responses.prompt = new_prompt_responses

    def _save(self, aggregated: dict[int, PassageResult], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = [aggregated[idx].to_dict() for idx in sorted(aggregated)]
        with tempfile.NamedTemporaryFile(
            "w",
            dir=output_path.parent,
            delete=False,
            suffix=".tmp",
            encoding="utf-8",
        ) as tmp:
            json.dump(ordered, tmp, indent=2, ensure_ascii=False)
            tmp_path = tmp.name
        os.replace(tmp_path, output_path)

    def _dataset_length(self, dataset_path: Path) -> int:
        with open(dataset_path) as f:
            return len(json.load(f))

    def _build_coverages(
        self,
        aggregated: dict[int, PassageResult],
        total: int,
    ) -> list[VariantCoverage]:
        coverages = []
        for variant in VARIANTS:
            done_indices = {
                idx
                for idx, result in aggregated.items()
                if result.scores.get(variant) is not None and 0 <= idx < total
            }
            missing = sorted(set(range(total)) - done_indices)
            coverages.append(
                VariantCoverage(
                    variant=variant,
                    done=len(done_indices),
                    total=total,
                    missing_indices=missing,
                )
            )
        return coverages
