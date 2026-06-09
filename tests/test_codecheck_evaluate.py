import math
from codecheck.models import CodeResult
from codecheck.evaluate import auc_pr_detect_incorrect


def _r(score, correct):
    return CodeResult("t", score, correct, "m", ["s"])


def test_perfect_separation_scores_one():
    results = [_r(0.9, False), _r(0.8, False), _r(0.1, True), _r(0.0, True)]
    assert auc_pr_detect_incorrect(results) == 1.0


def test_single_class_is_nan():
    results = [_r(0.5, True), _r(0.4, True)]
    assert math.isnan(auc_pr_detect_incorrect(results))
