from __future__ import annotations
from sklearn.metrics import auc, precision_recall_curve

from codecheck.models import CodeResult


def auc_pr_detect_incorrect(results: list[CodeResult]) -> float:
    y_true = [0 if r.is_correct else 1 for r in results]
    scores = [r.exec_score for r in results]
    if len(set(y_true)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return float(auc(recall, precision))
