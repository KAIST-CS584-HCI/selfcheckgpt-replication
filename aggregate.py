from __future__ import annotations

import argparse
from pathlib import Path

from replication.aggregate import AggregateConfig, AggregateRunner
from replication.score.base import DEFAULT_RESULTS_ROOT


DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_ROOT / "aggregated.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate per-slice score outputs into a single JSON, "
                    "and report per-variant coverage against the original dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the original dataset JSON (defines total passage count).",
    )
    parser.add_argument(
        "--inputs-dir",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT),
        help="Directory to scan for {method}-{start}-to-{end}.json slice files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Aggregated JSON destination (reused/extended if it already exists).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = AggregateConfig(
        dataset_path=Path(args.dataset),
        inputs_dir=Path(args.inputs_dir),
        output_path=Path(args.output),
    )
    report = AggregateRunner().run(config)
    print(report.format(), end="")


if __name__ == "__main__":
    main()
