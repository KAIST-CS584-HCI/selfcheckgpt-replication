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
    is mutually exclusive with `--limit`/`--random`/`--seed`; a bare run (no `--limit`)
    selects the **entire dataset** (limit=None). Raises ValueError on a conflicting
    combination."""
    if args.index is not None:
        if args.limit is not None or args.randomize or args.seed is not None:
            raise ValueError("--index cannot be combined with --limit, --random, or --seed")
        return None, args.index
    return args.limit, None   # None -> whole dataset


def _dataset_config(dataset: str):
    """Resolve the per-dataset pieces a run needs: (loader, batch harness, generation
    prompt builder, judge template). MBPP+ and HumanEval+ are function-call (shared harness,
    different judge wording); CodeHaluEval is whole-program stdin->stdout (its own harness and
    prompt builder)."""
    from codecheck.dataset import load_mbpp_plus, load_human_eval_plus, load_codehalu_eval
    from codecheck.generation.generator import build_prompt, build_stdio_prompt
    from codecheck.score.prompt import JUDGE_TEMPLATE, HUMANEVAL_JUDGE_TEMPLATE, CODEHALU_JUDGE_TEMPLATE
    from codecheck.execution.sandbox import run_batch_in_subprocess
    from codecheck.execution.stdio_sandbox import run_stdio_batch_in_subprocess

    if dataset == "codehalu":
        return load_codehalu_eval, run_stdio_batch_in_subprocess, build_stdio_prompt, CODEHALU_JUDGE_TEMPLATE
    if dataset == "humaneval":
        # HumanEval+ uses a sharper, divergence-seeking judge prompt (still oracle-free).
        return load_human_eval_plus, run_batch_in_subprocess, build_prompt, HUMANEVAL_JUDGE_TEMPLATE
    # MBPP+ keeps the original consistency prompt so its committed numbers stay comparable.
    return load_mbpp_plus, run_batch_in_subprocess, build_prompt, JUDGE_TEMPLATE


def _cmd_run(args: argparse.Namespace) -> None:
    from openai import AuthenticationError, OpenAI
    from codecheck.generation.generator import CodeGenerator
    from codecheck.score.prompt import PromptJudge
    from codecheck.score.ast import ASTScorer
    from codecheck.score.codebert import CodeBERTScorer, torch_available
    from codecheck.pipeline import run_dataset, load_results, append_result

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit("error: missing OPENROUTER_API_KEY (see .env.example)")
    model = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.5-9b").strip()
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    print(f"Model: {model}  (set OPENROUTER_MODEL to change)")

    load, harness, prompt_builder, judge_template = _dataset_config(args.dataset)
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=0)
    # CodeHaluEval is whole-program competitive (Codeforces-style) code, far harder than a
    # single function, so generation reasons by default to get usable programs; --think still
    # forces it on for the other datasets.
    gen_think = args.think or args.dataset == "codehalu"
    generator = CodeGenerator(client, model=model, think=gen_think, prompt_builder=prompt_builder)

    if args.method == "all":
        methods = {"exec", "prompt", "ast", "code_bert"}
    else:
        methods = {args.method}
    if "code_bert" in methods and not torch_available():
        sys.exit("error: method 'code_bert' requires torch — pip install torch")
    judge = (PromptJudge(client, model=model, think=args.think, template=judge_template)
             if "prompt" in methods else None)
    ast_scorer = ASTScorer(metric=args.ast_metric) if "ast" in methods else None
    codebert_scorer = CodeBERTScorer() if "code_bert" in methods else None

    try:
        limit, index = _resolve_selection(args)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    _setup_run_logging(verbose=args.verbose)
    try:
        problems = load(limit=limit, randomize=args.randomize, seed=args.seed, index=index)
    except IndexError as exc:
        sys.exit(f"error: {exc}")
    # Auto-resume: if the output file exists, skip problems already recorded (by task_id)
    # and add only the remainder.
    done_count = 0
    if Path(args.output).exists():
        try:
            existing = load_results(args.output)
        except ValueError as exc:
            sys.exit(f"error: {exc}")
        done = {r.task_id for r in existing}
        problems = [p for p in problems if p.task_id not in done]
        done_count = len(done)
        print(f"resuming: {done_count} done, {len(problems)} remaining in {args.output}")

    ast_note = f", ast-metric={args.ast_metric}" if ast_scorer is not None else ""
    print(f"Running methods={sorted(methods)} on {len(problems)} {args.dataset} problems "
          f"(n={args.n}, timeout={args.timeout}s, model={model}{ast_note})")
    if not problems:
        print("nothing to do — all selected problems already in the output file")
        return
    try:
        results = run_dataset(problems, generator, harness,
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
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    print(format_evaluation(results))


def _cmd_codebert(args: argparse.Namespace) -> None:
    """Offline: add a `code_bert` consistency score to an existing results file, reusing
    the stored main_code + sample_codes (no regeneration, no API). Rewrites in place. By
    default only results lacking a code_bert score are filled (resume a partial pass);
    --recompute rescores every result."""
    from codecheck.score.codebert import CodeBERTScorer, torch_available
    from codecheck.pipeline import load_results, save_results

    if not torch_available():
        sys.exit("error: 'codebert' requires torch — pip install torch")
    try:
        results = load_results(args.results)
    except FileNotFoundError:
        sys.exit(f"error: results file not found: {args.results}")
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    todo = results if args.recompute else [r for r in results if "code_bert" not in r.scores]
    scorer = CodeBERTScorer()
    for r in todo:
        r.scores["code_bert"] = scorer.score(r.main_code, r.sample_codes)
    save_results(results, args.results)
    skipped = len(results) - len(todo)
    note = "" if args.recompute else f" ({skipped} already scored, skipped)"
    print(f"Scored code_bert on {len(todo)} results in {args.results}{note}")


def _judge_template_for(task_id: str, override, default_tmpl, humaneval_tmpl):
    """Pick the judge template for one result. `override` (the --dataset value) forces a
    template; otherwise auto-detect from the task_id prefix (HumanEval/* uses the sharper
    divergence-seeking prompt, everything else the default consistency prompt)."""
    if override == "humaneval":
        return humaneval_tmpl
    if override == "mbpp":
        return default_tmpl
    return humaneval_tmpl if str(task_id).startswith("HumanEval/") else default_tmpl


def _cmd_prompt(args: argparse.Namespace) -> None:
    """Re-score the `prompt` (LLM-judge) variant on an existing results file, reusing the
    stored main_code + sample_codes (no regeneration). Unlike `codebert` this calls the API.
    The judge template is chosen per result (override via --dataset, else by task_id prefix)
    so the new HumanEval prompt applies to HumanEval results without touching MBPP+ ones.
    Rewrites in place."""
    from openai import OpenAI
    from tqdm import tqdm
    from codecheck.score.prompt import PromptJudge, JUDGE_TEMPLATE, HUMANEVAL_JUDGE_TEMPLATE
    from codecheck.pipeline import load_results, save_results

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit("error: missing OPENROUTER_API_KEY (see .env.example)")
    model = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.5-9b").strip()
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

    try:
        results = load_results(args.results)
    except FileNotFoundError:
        sys.exit(f"error: results file not found: {args.results}")
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=0)
    # One judge per distinct template; reuse it across the results that map to it.
    judges: dict[str, object] = {}
    # Re-scoring is API-bound (one judge round-trip per sample, ~minutes for a full file), so
    # show a live bar and save after each result — a Ctrl-C keeps the scores done so far.
    with tqdm(results, desc="prompt") as bar:
        for r in bar:
            tmpl = _judge_template_for(r.task_id, args.dataset, JUDGE_TEMPLATE, HUMANEVAL_JUDGE_TEMPLATE)
            if tmpl not in judges:
                judges[tmpl] = PromptJudge(client, model=model, template=tmpl)
            r.scores["prompt"] = judges[tmpl].score(r.main_code, r.sample_codes)
            save_results(results, args.results)   # incremental: never lose finished scores
    parse_failures = sum(getattr(j, "parse_failures", 0) for j in judges.values())
    print(f"Re-scored prompt on {len(results)} results in {args.results}")
    print(f"Judge parse failures: {parse_failures}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SelfCheck for code (Exec, Prompt, AST, CodeBERT) on MBPP+.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="generate, score, and save")
    run_p.add_argument("--limit", type=int, default=None,
                       help="number of problems to use (default: the entire dataset)")
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
    run_p.add_argument("--dataset", choices=["mbpp", "humaneval", "codehalu"], default="mbpp",
                       help="dataset: mbpp (MBPP+, default), humaneval (HumanEval+), or "
                            "codehalu (CodeHaluEval stdin/stdout)")
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
    cb_p.add_argument("--recompute", action="store_true",
                      help="rescore every result; default only fills results lacking a code_bert score")
    cb_p.set_defaults(func=_cmd_codebert)

    pr_p = sub.add_parser("prompt", help="re-score the prompt (LLM-judge) variant on an existing results file")
    pr_p.add_argument("--results", type=str, default=str(DEFAULT_OUTPUT),
                      help="results JSON path to augment in place")
    pr_p.add_argument("--dataset", choices=["mbpp", "humaneval"], default=None,
                      help="force a judge template; default auto-detects per result from the task_id prefix")
    pr_p.set_defaults(func=_cmd_prompt)
    return parser


def main(argv: list[str] | None = None) -> None:
    load_environment()
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
