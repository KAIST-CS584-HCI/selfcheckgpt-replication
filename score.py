import argparse
import json
import os
from pathlib import Path

from replication.score.base import DEFAULT_RESULTS_ROOT, REPO_ROOT, ScoreIO, ScoreRunner


METHODS = ("bert", "nli", "prompt")
DEFAULT_DATASET_PATHS = {
    "bert": REPO_ROOT / "data" / "dataset-generated.json",
    "nli": REPO_ROOT / "data" / "dataset-generated.json",
    "prompt": REPO_ROOT / "data" / "dataset-original.json",
}


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_environment(env_path: Path | None = None) -> None:
    env_path = env_path or REPO_ROOT / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_file(env_path)
    else:
        load_dotenv(env_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SelfCheckGPT replication scoring.")
    subparsers = parser.add_subparsers(dest="method", required=True)

    for method in METHODS:
        subparser = subparsers.add_parser(method, help=f"run {method} scoring")
        subparser.add_argument("--start", type=int, required=True, help="start index (inclusive)")
        subparser.add_argument(
            "--end",
            type=int,
            default=None,
            help="end index (exclusive); defaults to len(dataset)",
        )
        subparser.add_argument("--dataset", type=str, default=None, help="dataset JSON path")
        subparser.add_argument("--output", type=str, default=None, help="aggregate result JSON path")
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


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    return Path(args.dataset) if args.dataset is not None else DEFAULT_DATASET_PATHS[args.method]


def peek_dataset_length(dataset_path: Path) -> int:
    with open(dataset_path) as f:
        return len(json.load(f))


def build_score_io(args: argparse.Namespace, end: int) -> ScoreIO:
    dataset_path = resolve_dataset_path(args)
    output_path = Path(args.output) if args.output is not None else default_output_path(
        method=args.method,
        start=args.start,
        end=end,
    )
    return ScoreIO(dataset_path=dataset_path, output_path=output_path)


def default_output_path(method: str, start: int, end: int) -> Path:
    return DEFAULT_RESULTS_ROOT / f"{method}-{start}-to-{end}.json"


def main(argv: list[str] | None = None) -> None:
    load_environment()
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset_path = resolve_dataset_path(args)
    end = args.end if args.end is not None else peek_dataset_length(dataset_path)

    score_io = build_score_io(args, end=end)
    scorer = build_scorer(args)
    ScoreRunner(scorer, score_io).run(
        start=args.start,
        end=end,
    )


if __name__ == "__main__":
    main()
