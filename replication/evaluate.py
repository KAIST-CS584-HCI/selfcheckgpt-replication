"""
Unified evaluation for SelfCheckGPT replication.

Reproduces Table 2 of Manakul et al. (2023):
  - Sentence-level AUC-PR: NonFact, NonFact*, Factual
  - Passage-level correlations: Pearson, Spearman

Supports Prompt, BERT, and NLI variants. Pass --variant to select which to
evaluate; defaults to all variants present in the results file.

Label mapping (from paper §6):
  accurate         → 0.0 (factual)
  minor_inaccurate → 0.5 (non-factual)
  major_inaccurate → 1.0 (non-factual, hallucinated)

NonFact*  = major-inaccurate detection restricted to passages that are NOT
            total hallucination (passage-mean label < 1.0).

Run:
    python3 -m replication.evaluate --results <path> [--variant {prompt,bert,nli,all}]
"""
import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import auc, precision_recall_curve

from replication.entity import PassageResult

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR  = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(RESULTS_DIR, "results.json")

LABEL_SCORE = {
    "accurate":         0.0,
    "minor_inaccurate": 0.5,
    "major_inaccurate": 1.0,
}

VARIANTS = ["prompt", "bert", "nli"]

PAPER_BASELINES = {
    "bert":   {"nonfact": 81.96, "nonfact_star": 45.96, "factual": 44.23, "pearson": 58.18, "spearman": 55.90},
    "nli":    {"nonfact": 92.50, "nonfact_star": 45.17, "factual": 66.08, "pearson": 74.14, "spearman": 73.78},
    "prompt": {"nonfact": 93.42, "nonfact_star": 53.19, "factual": 67.09, "pearson": 78.32, "spearman": 78.30},
}

PAPER_LABELS = {
    "bert":   "SelfCk-BERTScore (paper, ChatGPT)",
    "nli":    "SelfCk-NLI       (paper, ChatGPT)",
    "prompt": "SelfCk-Prompt    (paper, ChatGPT)",
}

VARIANT_LABELS = {
    "bert":   "SelfCk-BERTScore",
    "nli":    "SelfCk-NLI      ",
    "prompt": "SelfCk-Prompt   ",
}

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(path: str) -> list[PassageResult]:
    with open(path) as f:
        return [PassageResult.from_dict(d) for d in json.load(f)]


def scores_for_variant(result: PassageResult, variant: str) -> list[float] | None:
    return {
        "prompt": result.prompt_scores,
        "bert":   result.bert_scores,
        "nli":    result.nli_scores,
    }[variant]


def active_variants(results: list[PassageResult], requested: list[str]) -> list[str]:
    """Return variants that have scores present in at least one result."""
    return [v for v in requested if any(scores_for_variant(r, v) is not None for r in results)]


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------

@dataclass
class Flattened:
    scores:        np.ndarray
    labels:        np.ndarray
    passage_mean:  np.ndarray


def flatten(results: list[PassageResult], variant: str) -> Flattened:
    scores, labels, passage_mean = [], [], []
    for r in results:
        sent_scores = scores_for_variant(r, variant)
        if sent_scores is None:
            continue
        gold = np.array([LABEL_SCORE[a] for a in r.annotation])
        pm   = float(gold.mean())
        scores.extend(sent_scores)
        labels.extend(gold.tolist())
        passage_mean.extend([pm] * len(sent_scores))
    return Flattened(
        scores       = np.asarray(scores),
        labels       = np.asarray(labels),
        passage_mean = np.asarray(passage_mean),
    )


# ---------------------------------------------------------------------------
# Sentence-level AUC-PR
# ---------------------------------------------------------------------------

def auc_pr_nonfact(f: Flattened) -> float:
    y_true = (f.labels > 0).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, f.scores)
    return float(auc(recall, precision))


def auc_pr_nonfact_star(f: Flattened) -> float:
    mask = f.passage_mean < 1.0
    y_true = (f.labels[mask] == 1.0).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, f.scores[mask])
    return float(auc(recall, precision))


def auc_pr_factual(f: Flattened) -> float:
    y_true = (f.labels == 0.0).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, -f.scores)
    return float(auc(recall, precision))


# ---------------------------------------------------------------------------
# Passage-level correlations
# ---------------------------------------------------------------------------

def passage_correlations(results: list[PassageResult], variant: str) -> tuple[float, float]:
    pred, gold = [], []
    for r in results:
        sent_scores = scores_for_variant(r, variant)
        if sent_scores is None:
            continue
        pred.append(float(np.mean(sent_scores)))
        gold.append(float(np.mean([LABEL_SCORE[a] for a in r.annotation])))
    pearson  = float(pearsonr(pred, gold).statistic)
    spearman = float(spearmanr(pred, gold).statistic)
    return pearson, spearman


# ---------------------------------------------------------------------------
# Evaluate one variant
# ---------------------------------------------------------------------------

def evaluate(results: list[PassageResult], variant: str) -> dict:
    f = flatten(results, variant)
    pearson, spearman = passage_correlations(results, variant)
    return {
        "variant":      variant,
        "n_passages":   len([r for r in results if scores_for_variant(r, variant) is not None]),
        "n_sentences":  len(f.scores),
        "nonfact":      auc_pr_nonfact(f) * 100,
        "nonfact_star": auc_pr_nonfact_star(f) * 100,
        "factual":      auc_pr_factual(f) * 100,
        "pearson":      pearson * 100,
        "spearman":     spearman * 100,
    }


# ---------------------------------------------------------------------------
# Response distribution (prompt-only)
# ---------------------------------------------------------------------------

def response_distribution(results: list[PassageResult]) -> dict:
    counts: dict[str, int] = {}
    total = 0
    for r in results:
        for sent_responses in r.prompt_responses:
            for msg in sent_responses:
                total += 1
                if msg is None:
                    key = "null"
                else:
                    text = msg.lower().strip()
                    if text[:3] == "yes":
                        key = "Yes"
                    elif text[:2] == "no":
                        key = "No"
                    else:
                        key = msg.strip() if msg.strip() else "(empty)"
                counts[key] = counts.get(key, 0) + 1
    return {"total": total, "counts": counts}


def print_response_distribution(dist: dict) -> None:
    total = dist["total"]
    counts = dist["counts"]
    if total == 0:
        return
    yes = counts.get("Yes", 0)
    no  = counts.get("No",  0)
    yn  = yes + no
    other_counts = {k: v for k, v in counts.items() if k not in ("Yes", "No")}
    print(f"Response distribution (N={total:,} total API calls):")
    print(f"  Yes   : {yes:>6,}  ({100 * yes / total:5.1f}%)")
    print(f"  No    : {no:>6,}  ({100 * no  / total:5.1f}%)")
    print(f"  Yes+No: {yn:>6,}  ({100 * yn  / total:5.1f}%)")
    if other_counts:
        print(f"  Other :")
        for k, v in sorted(other_counts.items(), key=lambda x: -x[1]):
            print(f"    {k!r:<20} {v:>6,}  ({100 * v / total:5.1f}%)")
    print()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(metrics_list: list[dict], variants: list[str]) -> None:
    header = f"{'Method':<28}  {'NonFact':>8}  {'NonFact*':>8}  {'Factual':>8}  {'Pearson':>8}  {'Spearman':>8}"
    sep    = "-" * len(header)

    if metrics_list:
        first = metrics_list[0]
        print(f"Passages: {first['n_passages']} | Sentences: {first['n_sentences']}")
        print()

    print(header)
    print(sep)
    for m in metrics_list:
        label = VARIANT_LABELS[m["variant"]]
        print(
            f"{label:<28}  "
            f"{m['nonfact']:>8.2f}  "
            f"{m['nonfact_star']:>8.2f}  "
            f"{m['factual']:>8.2f}  "
            f"{m['pearson']:>8.2f}  "
            f"{m['spearman']:>8.2f}"
        )

    print(sep)
    print("Paper (Table 2, N=20):")
    for v in variants:
        b = PAPER_BASELINES[v]
        label = PAPER_LABELS[v]
        print(
            f"{label:<28}  "
            f"{b['nonfact']:>8.2f}  "
            f"{b['nonfact_star']:>8.2f}  "
            f"{b['factual']:>8.2f}  "
            f"{b['pearson']:>8.2f}  "
            f"{b['spearman']:>8.2f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="Evaluate SelfCheckGPT replication results.")
_parser.add_argument("--results", type=str, default=RESULTS_PATH, help="Path to results JSON file")
_parser.add_argument("--variant", type=str, default="all",
                     choices=["prompt", "bert", "nli", "all"],
                     help="Which score variant(s) to evaluate (default: all present)")
_parser.add_argument("--start", type=int, default=None, help="First index to include (inclusive)")
_parser.add_argument("--end",   type=int, default=None, help="Last index to include (inclusive)")


def main() -> None:
    args    = _parser.parse_args()
    results = load_results(args.results)

    start = args.start if args.start is not None else 0
    end   = args.end   if args.end   is not None else len(results) - 1
    results = results[start:end + 1]

    requested = VARIANTS if args.variant == "all" else [args.variant]
    variants  = active_variants(results, requested)

    if not variants:
        print("No score data found for the requested variant(s).")
        return

    if "prompt" in variants:
        dist = response_distribution(results)
        print_response_distribution(dist)

    metrics_list = [evaluate(results, v) for v in variants]
    print_summary(metrics_list, variants)


if __name__ == "__main__":
    main()
