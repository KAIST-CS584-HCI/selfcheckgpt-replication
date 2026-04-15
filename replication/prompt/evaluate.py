"""
Evaluation for SelfCheckGPT-Prompt replication.

Reproduces Table 2 of Manakul et al. (2023):
  - Sentence-level AUC-PR: NonFact, NonFact*, Factual
  - Passage-level correlations: Pearson, Spearman

Label mapping (from paper §6):
  accurate         → 0.0 (factual)
  minor_inaccurate → 0.5 (non-factual)
  major_inaccurate → 1.0 (non-factual, hallucinated)

NonFact*  = major-inaccurate detection restricted to passages that are NOT
            total hallucination (passage-mean label < 1.0).

Run:
    python3 -m replication.prompt.evaluate [--results <path>]
"""
import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

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

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(path: str) -> list[PassageResult]:
    with open(path) as f:
        return [PassageResult.from_dict(d) for d in json.load(f)]


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------

@dataclass
class Flattened:
    """All sentences flattened across passages, with labels + scores."""
    scores:        np.ndarray  # predicted sent-level hallucination score, shape (N,)
    labels:        np.ndarray  # gold per-sentence score in {0.0, 0.5, 1.0}, shape (N,)
    passage_mean:  np.ndarray  # gold passage-mean label broadcast to each sentence


def flatten(results: list[PassageResult]) -> Flattened:
    scores, labels, passage_mean = [], [], []
    for r in results:
        gold = np.array([LABEL_SCORE[a] for a in r.annotation])
        pm   = float(gold.mean())
        scores.extend(r.sent_scores)
        labels.extend(gold.tolist())
        passage_mean.extend([pm] * len(r.sent_scores))
    return Flattened(
        scores       = np.asarray(scores),
        labels       = np.asarray(labels),
        passage_mean = np.asarray(passage_mean),
    )


# ---------------------------------------------------------------------------
# Sentence-level AUC-PR
# ---------------------------------------------------------------------------

def auc_pr_nonfact(f: Flattened) -> float:
    """Positive class = non-factual (label > 0)."""
    y_true = (f.labels > 0).astype(int)
    return float(average_precision_score(y_true, f.scores))


def auc_pr_nonfact_star(f: Flattened) -> float:
    """
    Positive class = major_inaccurate (label == 1.0),
    restricted to passages that are NOT total hallucination (passage_mean < 1.0).
    """
    mask = f.passage_mean < 1.0
    y_true = (f.labels[mask] == 1.0).astype(int)
    return float(average_precision_score(y_true, f.scores[mask]))


def auc_pr_factual(f: Flattened) -> float:
    """
    Positive class = factual (label == 0).
    Higher predicted score = more hallucinated, so we flip the sign.
    """
    y_true = (f.labels == 0.0).astype(int)
    return float(average_precision_score(y_true, -f.scores))


# ---------------------------------------------------------------------------
# Passage-level correlations
# ---------------------------------------------------------------------------

def passage_correlations(results: list[PassageResult]) -> tuple[float, float]:
    pred = np.array([float(np.mean(r.sent_scores)) for r in results])
    gold = np.array([float(np.mean([LABEL_SCORE[a] for a in r.annotation])) for r in results])
    pearson  = float(pearsonr(pred, gold).statistic)
    spearman = float(spearmanr(pred, gold).statistic)
    return pearson, spearman


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def evaluate(results: list[PassageResult]) -> dict:
    f = flatten(results)
    pearson, spearman = passage_correlations(results)
    return {
        "n_passages":   len(results),
        "n_sentences":  len(f.scores),
        "nonfact":      auc_pr_nonfact(f) * 100,
        "nonfact_star": auc_pr_nonfact_star(f) * 100,
        "factual":      auc_pr_factual(f) * 100,
        "pearson":      pearson * 100,
        "spearman":     spearman * 100,
    }


def print_summary(m: dict) -> None:
    print(f"Passages: {m['n_passages']} | Sentences: {m['n_sentences']}")
    print()
    print(f"{'Method':<20}  {'NonFact':>8}  {'NonFact*':>8}  {'Factual':>8}  {'Pearson':>8}  {'Spearman':>8}")
    print("-" * 72)
    print(
        f"{'SelfCk-Prompt':<20}  "
        f"{m['nonfact']:>8.2f}  "
        f"{m['nonfact_star']:>8.2f}  "
        f"{m['factual']:>8.2f}  "
        f"{m['pearson']:>8.2f}  "
        f"{m['spearman']:>8.2f}"
    )
    print()
    print("Paper (Table 2, SelfCk-Prompt w/ ChatGPT, N=20):")
    print(f"{'SelfCk-Prompt (paper)':<20}  {93.42:>8.2f}  {53.19:>8.2f}  {67.09:>8.2f}  {78.32:>8.2f}  {78.30:>8.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="Evaluate SelfCheckGPT-Prompt replication results.")
_parser.add_argument("--results", type=str, default=RESULTS_PATH, help="Path to results.json")


def main() -> None:
    args    = _parser.parse_args()
    results = load_results(args.results)
    metrics = evaluate(results)
    print_summary(metrics)


if __name__ == "__main__":
    main()
