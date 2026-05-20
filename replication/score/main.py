import argparse
from pathlib import Path

from replication.score.base import DEFAULT_RESULTS_ROOT, REPO_ROOT, ScoreIO, ScoreRunner


METHODS = ("bert", "nli", "prompt")
DEFAULT_DATASET_PATHS = {
    "bert": REPO_ROOT / "data" / "dataset-generated.json",
    "nli": REPO_ROOT / "data" / "dataset-generated.json",
    "prompt": REPO_ROOT / "data" / "dataset-original.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SelfCheckGPT replication scoring.")
    subparsers = parser.add_subparsers(dest="method", required=True)

    for method in METHODS:
        subparser = subparsers.add_parser(method, help=f"run {method} scoring")
        subparser.add_argument("--start", type=int, required=True, help="start index (inclusive)")
        subparser.add_argument("--end", type=int, required=True, help="end index (exclusive)")
        subparser.add_argument("--dataset", type=str, default=None, help="dataset JSON path")
        subparser.add_argument("--output", type=str, default=None, help="aggregate result JSON path")
        subparser.add_argument("--overwrite", action="store_true", help="overwrite existing aggregate output")
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


def build_score_io(args: argparse.Namespace) -> ScoreIO:
    dataset_path = Path(args.dataset) if args.dataset is not None else DEFAULT_DATASET_PATHS[args.method]
    output_path = Path(args.output) if args.output is not None else DEFAULT_RESULTS_ROOT / f"{args.method}.json"
    return ScoreIO(dataset_path=dataset_path, output_path=output_path, overwrite=args.overwrite)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    score_io = build_score_io(args)
    scorer = build_scorer(args)
    ScoreRunner(scorer, score_io).run(
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
