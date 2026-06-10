import math
from codecheck.models import CodeResult
from codecheck.evaluate import (
    auc_pr_detect_incorrect, prevalence_baseline, score_histogram,
)


def _r(score, correct, method="exec"):
    return CodeResult("t", {method: score}, correct, "m", ["s"])


def test_perfect_separation_scores_one():
    results = [_r(0.9, False), _r(0.8, False), _r(0.1, True), _r(0.0, True)]
    assert auc_pr_detect_incorrect(results) == 1.0


def test_single_class_is_nan():
    results = [_r(0.5, True), _r(0.4, True)]
    assert math.isnan(auc_pr_detect_incorrect(results))


def test_auc_selects_named_method():
    # exec ranks perfectly; prompt ranks inverted on the same results
    results = [
        CodeResult("a", {"exec": 0.9, "prompt": 0.0}, False, "m", []),
        CodeResult("b", {"exec": 0.1, "prompt": 1.0}, True, "m", []),
    ]
    assert auc_pr_detect_incorrect(results, method="exec") == 1.0
    assert auc_pr_detect_incorrect(results, method="prompt") < 1.0


def test_prevalence_baseline_is_positive_fraction():
    results = [_r(0.5, False), _r(0.5, False), _r(0.5, True), _r(0.5, True)]
    assert prevalence_baseline(results) == 0.5


def test_score_histogram_splits_by_class():
    results = [_r(0.05, True), _r(0.05, True), _r(0.95, False)]
    hist = score_histogram(results, method="exec", bins=10)
    assert len(hist) == 10
    first = hist[0]      # bucket [0.0, 0.1)
    last = hist[-1]      # bucket [0.9, 1.0]
    assert first["correct"] == 2 and first["incorrect"] == 0
    assert last["incorrect"] == 1 and last["correct"] == 0
