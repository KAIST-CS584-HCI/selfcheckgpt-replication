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


def _setup_run_logging(verbose: bool = False) -> None:
    handler = _TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    root = logging.getLogger("codecheck")
    # INFO keeps a normal run quiet apart from the per-call anomaly warnings (truncated /
    # empty responses); --verbose drops to DEBUG for the per-call latency/finish/token detail.
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers = [handler]
    root.propagate = False


def _resolve_selection(args: argparse.Namespace) -> tuple[int | None, int | None]:
    """Resolve problem selection into (limit, index). `--index` runs a single problem and
    is mutually exclusive with `--limit`/`--random`/`--seed`; a bare run defaults to the
    first 20. Raises ValueError on a conflicting combination."""
    if args.index is not None:
        if args.limit is not None or args.randomize or args.seed is not None:
            raise ValueError("--index cannot be combined with --limit, --random, or --seed")
        return None, args.index
    return (args.limit if args.limit is not None else 20), None


def _cmd_run(args: argparse.Namespace) -> None:
    from openai import AuthenticationError, OpenAI
    from codecheck.dataset import load_mbpp_plus
    from codecheck.generation import CodeGenerator
    from codecheck.prompt_score import PromptJudge
    from codecheck.ast_score import ASTScorer
    from codecheck.codebert_score import CodeBERTScorer, torch_available
    from codecheck.execution import run_batch_in_subprocess
    from codecheck.pipeline import run_dataset, load_results, append_result

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit("error: missing OPENROUTER_API_KEY (see .env.example)")
    model = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.5-9b").strip()
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    print(f"Model: {model}  (set OPENROUTER_MODEL to change)")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=0)
    generator = CodeGenerator(client, model=model, think=args.think)

    if args.method == "all":
        methods = {"exec", "prompt", "ast", "code_bert"}
    else:
        methods = {args.method}
    if "code_bert" in methods and not torch_available():
        sys.exit("error: method 'code_bert' requires torch — pip install torch")
    judge = PromptJudge(client, model=model, think=args.think) if "prompt" in methods else None
    ast_scorer = ASTScorer(metric=args.ast_metric) if "ast" in methods else None
    codebert_scorer = CodeBERTScorer() if "code_bert" in methods else None

    try:
        limit, index = _resolve_selection(args)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    _setup_run_logging(verbose=args.verbose)
    try:
        problems = load_mbpp_plus(limit=limit, randomize=args.randomize, seed=args.seed, index=index)
    except IndexError as exc:
        sys.exit(f"error: {exc}")
    # Auto-resume: if the output file exists, skip problems already recorded (by task_id)
    # and add only the remainder.
    done_count = 0
    if Path(args.output).exists():
        existing = load_results(args.output)
        done = {r.task_id for r in existing}
        problems = [p for p in problems if p.task_id not in done]
        done_count = len(done)
        print(f"resuming: {done_count} done, {len(problems)} remaining in {args.output}")

    ast_note = f", ast-metric={args.ast_metric}" if ast_scorer is not None else ""
    print(f"Running methods={sorted(methods)} on {len(problems)} problems "
          f"(n={args.n}, timeout={args.timeout}s, model={model}{ast_note})")
    if not problems:
        print("nothing to do — all selected problems already in the output file")
        return
    try:
        results = run_dataset(problems, generator, run_batch_in_subprocess,
                              n_samples=args.n, timeout=args.timeout,
                              methods=methods, judge=judge, ast_scorer=ast_scorer,
                              codebert_scorer=codebert_scorer,
                              on_result=lambda r: append_result(r, args.output))
    except AuthenticationError:
        sys.exit("error: OpenRouter rejected OPENROUTER_API_KEY (expects an sk-or-v1-… key; see .env.example)")
    print(f"Saved {done_count + len(results)} results to {args.output}")
    if judge is not None:
        print(f"Judge parse failures: {judge.parse_failures}")
    if ast_scorer is not None:
        print(f"AST parse failures: {ast_scorer.parse_failures}")


def _methods_present(results) -> list[str]:
    seen: list[str] = []
    for r in results:
        for m in r.scores:
            if m not in seen:
                seen.append(m)
    return seen


def format_evaluation(results) -> str:
    from codecheck.evaluate import (
        auc_pr_detect_incorrect, auc_pr_detect_correct, score_label_correlation,
        prevalence_baseline, score_histogram,
    )
    n = len(results)
    n_incorrect = sum(1 for r in results if not r.is_correct)
    lines = [f"n={n}  incorrect={n_incorrect}  correct={n - n_incorrect}",
             f"baseline (incorrect prevalence): {prevalence_baseline(results):.4f}", ""]
    for method in _methods_present(results):
        method_results = [r for r in results if method in r.scores]
        auc_inc = auc_pr_detect_incorrect(method_results, method=method)
        auc_cor = auc_pr_detect_correct(method_results, method=method)
        pearson, spearman = score_label_correlation(method_results, method=method)
        lines.append(f"[{method}] AUC-PR detect-incorrect: {auc_inc:.4f}  "
                     f"detect-correct: {auc_cor:.4f}  (n={len(method_results)})")
        lines.append(f"  corr(score, incorrect): pearson={pearson:.4f}  spearman={spearman:.4f}")
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


def _cmd_codebert(args: argparse.Namespace) -> None:
    """Offline: add a `code_bert` consistency score to an existing results file, reusing
    the stored main_code + sample_codes (no regeneration, no API). Rewrites in place."""
    from codecheck.codebert_score import CodeBERTScorer, torch_available
    from codecheck.pipeline import load_results, save_results

    if not torch_available():
        sys.exit("error: 'codebert' requires torch — pip install torch")
    try:
        results = load_results(args.results)
    except FileNotFoundError:
        sys.exit(f"error: results file not found: {args.results}")
    scorer = CodeBERTScorer()
    for r in results:
        r.scores["code_bert"] = scorer.score(r.main_code, r.sample_codes)
    save_results(results, args.results)
    print(f"Added code_bert to {len(results)} results in {args.results}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SelfCheck for code (Exec, Prompt, AST, CodeBERT) on MBPP+.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="generate, score, and save")
    run_p.add_argument("--limit", type=int, default=None, help="number of MBPP+ problems (default 20)")
    run_p.add_argument("--index", type=int, default=None,
                       help="run only the single problem at this 0-based dataset position; "
                            "cannot be combined with --limit/--random/--seed")
    run_p.add_argument("--n", type=int, default=5, help="samples per problem (T=1)")
    run_p.add_argument("--timeout", type=float, default=5.0, help="per-call execution timeout (s)")
    run_p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="results JSON path")
    run_p.add_argument("--random", dest="randomize", action="store_true",
                       help="take a random sample of --limit problems instead of the first --limit in dataset order")
    run_p.add_argument("--seed", type=int, default=None, help="random seed for a reproducible sample")
    run_p.add_argument("--think", action="store_true",
                       help="enable model chain-of-thought reasoning (much slower; default off)")
    run_p.add_argument("-v", "--verbose", action="store_true",
                       help="log per-call API detail (latency, finish_reason, completion tokens) at DEBUG")
    run_p.add_argument("--method", choices=["exec", "prompt", "ast", "code_bert", "all"], default="exec",
                       help="which consistency scorer(s) to run")
    run_p.add_argument("--ast-metric", choices=["jaccard", "ted"], default="jaccard",
                       help="AST structural metric: jaccard (bag-of-node-types, default) or "
                            "ted (tree edit distance). Only used when --method includes ast")
    run_p.set_defaults(func=_cmd_run, randomize=False)

    eval_p = sub.add_parser("evaluate", help="report AUC-PR from a results file")
    eval_p.add_argument("--results", type=str, default=str(DEFAULT_OUTPUT), help="results JSON path")
    eval_p.set_defaults(func=_cmd_evaluate)

    cb_p = sub.add_parser("codebert", help="add a code_bert score to an existing results file (offline)")
    cb_p.add_argument("--results", type=str, default=str(DEFAULT_OUTPUT),
                      help="results JSON path to augment in place")
    cb_p.set_defaults(func=_cmd_codebert)
    return parser


def main(argv: list[str] | None = None) -> None:
    load_environment()
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
