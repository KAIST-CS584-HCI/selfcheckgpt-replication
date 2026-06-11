from __future__ import annotations
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import auc, precision_recall_curve

from codecheck.models import CodeResult


def auc_pr_detect_incorrect(results: list[CodeResult], method: str = "exec") -> float:
    """AUC-PR for detecting the INCORRECT class (the replication's `nonfact`): positive =
    incorrect, score used as-is (higher = more likely incorrect)."""
    y_true = [0 if r.is_correct else 1 for r in results]
    scores = [r.scores[method] for r in results]
    if len(set(y_true)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return float(auc(recall, precision))


def auc_pr_detect_correct(results: list[CodeResult], method: str = "exec") -> float:
    """AUC-PR for detecting the CORRECT class (the replication's `factual`): positive =
    correct, scores negated so a lower inconsistency score ranks as more likely correct.
    Same trapezoidal precision_recall_curve + auc as the incorrect-detection metric."""
    y_true = [1 if r.is_correct else 0 for r in results]
    scores = [-r.scores[method] for r in results]
    if len(set(y_true)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return float(auc(recall, precision))


def score_label_correlation(results: list[CodeResult], method: str = "exec") -> tuple[float, float]:
    """(pearson, spearman) of the consistency score vs incorrectness (1 = incorrect).
    Positive = the score tracks hallucination. NaN if either side is constant (undefined)."""
    scores = [r.scores[method] for r in results]
    labels = [0 if r.is_correct else 1 for r in results]
    if len(set(scores)) < 2 or len(set(labels)) < 2:
        return float("nan"), float("nan")
    return float(pearsonr(scores, labels).statistic), float(spearmanr(scores, labels).statistic)


def prevalence_baseline(results: list[CodeResult]) -> float:
    """PR no-skill floor = fraction of incorrect (positive) mains."""
    if not results:
        return float("nan")
    return sum(1 for r in results if not r.is_correct) / len(results)


def score_histogram(results: list[CodeResult], method: str = "exec", bins: int = 10) -> list[dict]:
    """Per-bin counts split by true class. Bin i covers [i/bins, (i+1)/bins);
    the final bin is closed on the right so score == 1.0 lands in it."""
    hist = [{"lo": i / bins, "hi": (i + 1) / bins, "correct": 0, "incorrect": 0}
            for i in range(bins)]
    for r in results:
        s = r.scores[method]
        idx = min(int(s * bins), bins - 1)
        key = "correct" if r.is_correct else "incorrect"
        hist[idx][key] += 1
    return hist
