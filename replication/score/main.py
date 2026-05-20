import argparse
from pathlib import Path

from replication.score.base import ScoreRunner


METHODS = ("bert", "nli", "prompt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SelfCheckGPT replication scoring.")
    subparsers = parser.add_subparsers(dest="method", required=True)

    for method in METHODS:
        subparser = subparsers.add_parser(method, help=f"run {method} scoring")
        subparser.add_argument("--start", type=int, required=True, help="start index (inclusive)")
        subparser.add_argument("--end", type=int, required=True, help="end index (exclusive)")
        subparser.add_argument("--dataset", type=str, default=None, help="dataset JSON path")
        subparser.add_argument("--output-dir", type=str, default=None, help="directory for per-index result JSON files")
        subparser.add_argument("--overwrite", action="store_true", help="overwrite existing per-index outputs")
        if method == "prompt":
            subparser.add_argument("--think", action="store_true", default=False, help="enable reasoning effort")

    return parser


def build_scorer(args: argparse.Namespace):
    if args.method == "bert":
        from replication.score.bert import BertScorer
        return BertScorer()
    if args.method == "nli":
        from replication.score.nli import NliScorer
        return NliScorer()
    if args.method == "prompt":
        from replication.score.prompt import PromptScorer
        return PromptScorer(think=args.think)
    raise ValueError(f"Unknown scoring method: {args.method}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    scorer = build_scorer(args)
    ScoreRunner(scorer).run(
        start=args.start,
        end=args.end,
        dataset_path=Path(args.dataset) if args.dataset is not None else None,
        output_dir=Path(args.output_dir) if args.output_dir is not None else None,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
