from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from score import load_environment  # reuse existing env loader

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = REPO_ROOT / "output" / "codecheck-exec.json"


def _cmd_run(args: argparse.Namespace) -> None:
    from openai import AuthenticationError, OpenAI
    from codecheck.dataset import load_mbpp_plus
    from codecheck.generation import CodeGenerator
    from codecheck.execution import run_batch_in_subprocess
    from codecheck.pipeline import run_dataset, save_results

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit("error: missing OPENROUTER_API_KEY (see .env.example)")
    model = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.5-9b").strip()
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=0)
    generator = CodeGenerator(client, model=model, think=args.think)

    problems = load_mbpp_plus(limit=args.limit, randomize=args.randomize, seed=args.seed)
    try:
        results = run_dataset(problems, generator, run_batch_in_subprocess, n_samples=args.n, timeout=args.timeout)
    except AuthenticationError:
        sys.exit("error: OpenRouter rejected OPENROUTER_API_KEY (expects an sk-or-v1-… key; see .env.example)")
    save_results(results, args.output)
    print(f"Saved {len(results)} results to {args.output}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from codecheck.pipeline import load_results
    from codecheck.evaluate import auc_pr_detect_incorrect

    try:
        results = load_results(args.results)
    except FileNotFoundError:
        sys.exit(f"error: results file not found: {args.results}")
    n_incorrect = sum(1 for r in results if not r.is_correct)
    auc_pr = auc_pr_detect_incorrect(results)
    print(f"AUC-PR (detect incorrect): {auc_pr:.4f}")
    print(f"n={len(results)}  incorrect={n_incorrect}  correct={len(results) - n_incorrect}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SelfCheck-Exec on MBPP+ (iteration 1).")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="generate, score, and save")
    run_p.add_argument("--limit", type=int, default=20, help="number of MBPP+ problems")
    run_p.add_argument("--n", type=int, default=5, help="samples per problem (T=1)")
    run_p.add_argument("--timeout", type=float, default=5.0, help="per-call execution timeout (s)")
    run_p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="results JSON path")
    run_p.add_argument("--no-random", dest="randomize", action="store_false",
                       help="use the first --limit problems in dataset order instead of a random sample")
    run_p.add_argument("--seed", type=int, default=None, help="random seed for a reproducible sample")
    run_p.add_argument("--think", action="store_true",
                       help="enable model chain-of-thought reasoning (much slower; default off)")
    run_p.set_defaults(func=_cmd_run, randomize=True)

    eval_p = sub.add_parser("evaluate", help="report AUC-PR from a results file")
    eval_p.add_argument("--results", type=str, default=str(DEFAULT_OUTPUT), help="results JSON path")
    eval_p.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    load_environment()
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
