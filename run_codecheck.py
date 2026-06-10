from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path

from score import load_environment  # reuse existing env loader

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = REPO_ROOT / "output" / "codecheck.json"


class _TqdmLoggingHandler(logging.Handler):
    """Route log records through tqdm.write so they don't corrupt the progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        from tqdm import tqdm
        try:
            tqdm.write(self.format(record))
        except Exception:  # noqa: BLE001 — never let logging crash a run
            self.handleError(record)


def _setup_run_logging() -> None:
    handler = _TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    root = logging.getLogger("codecheck")
    root.setLevel(logging.INFO)
    root.handlers = [handler]
    root.propagate = False


def _cmd_run(args: argparse.Namespace) -> None:
    from openai import AuthenticationError, OpenAI
    from codecheck.dataset import load_mbpp_plus
    from codecheck.generation import CodeGenerator
    from codecheck.prompt_score import PromptJudge
    from codecheck.execution import run_batch_in_subprocess
    from codecheck.pipeline import run_dataset, save_results

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit("error: missing OPENROUTER_API_KEY (see .env.example)")
    model = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.5-9b").strip()
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=0)
    generator = CodeGenerator(client, model=model, think=args.think)

    methods = {"exec", "prompt"} if args.method == "both" else {args.method}
    judge = PromptJudge(client, model=model, think=args.think) if "prompt" in methods else None

    _setup_run_logging()
    problems = load_mbpp_plus(limit=args.limit, randomize=args.randomize, seed=args.seed)
    print(f"Running methods={sorted(methods)} on {len(problems)} problems "
          f"(n={args.n}, timeout={args.timeout}s, model={model})")
    try:
        results = run_dataset(problems, generator, run_batch_in_subprocess,
                              n_samples=args.n, timeout=args.timeout,
                              methods=methods, judge=judge)
    except AuthenticationError:
        sys.exit("error: OpenRouter rejected OPENROUTER_API_KEY (expects an sk-or-v1-… key; see .env.example)")
    save_results(results, args.output)
    print(f"Saved {len(results)} results to {args.output}")
    if judge is not None:
        print(f"Judge parse failures: {judge.parse_failures}")


def _methods_present(results) -> list[str]:
    seen: list[str] = []
    for r in results:
        for m in r.scores:
            if m not in seen:
                seen.append(m)
    return seen


def format_evaluation(results) -> str:
    from codecheck.evaluate import (
        auc_pr_detect_incorrect, prevalence_baseline, score_histogram,
    )
    n = len(results)
    n_incorrect = sum(1 for r in results if not r.is_correct)
    lines = [f"n={n}  incorrect={n_incorrect}  correct={n - n_incorrect}",
             f"baseline (incorrect prevalence): {prevalence_baseline(results):.4f}", ""]
    for method in _methods_present(results):
        method_results = [r for r in results if method in r.scores]
        auc_pr = auc_pr_detect_incorrect(method_results, method=method)
        lines.append(f"[{method}] AUC-PR (detect incorrect): {auc_pr:.4f}  (n={len(method_results)})")
        lines.append(f"  {method} score histogram (bin: correct/incorrect):")
        for b in score_histogram(method_results, method=method, bins=10):
            lines.append(f"    [{b['lo']:.1f},{b['hi']:.1f}) {b['correct']}/{b['incorrect']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from codecheck.pipeline import load_results

    try:
        results = load_results(args.results)
    except FileNotFoundError:
        sys.exit(f"error: results file not found: {args.results}")
    print(format_evaluation(results))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SelfCheck for code (Exec + Prompt) on MBPP+.")
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
    run_p.add_argument("--method", choices=["exec", "prompt", "both"], default="exec",
                       help="which consistency scorer(s) to run")
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
