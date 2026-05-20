from __future__ import annotations

import argparse

from replication.evaluation.constants import RESULTS_PATH
from replication.evaluation.report import EvaluationReportRenderer
from replication.evaluation.runner import EvaluationConfig, EvaluationRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SelfCheckGPT replication results.")
    parser.add_argument("--output", type=str, default=str(RESULTS_PATH), help="Path to score output JSON file")
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["prompt", "bert", "nli", "all"],
        help="Which score variant(s) to evaluate (default: all present)",
    )
    parser.add_argument("--start", type=int, default=None, help="First index to include (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="Last index to include (inclusive)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = EvaluationRunner().run(
        EvaluationConfig(
            results_path=args.output,
            variant=args.variant,
            start=args.start,
            end=args.end,
        )
    )
    print(EvaluationReportRenderer().render(report), end="")


if __name__ == "__main__":
    main()
