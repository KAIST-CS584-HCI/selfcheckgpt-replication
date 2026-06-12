import os

import numpy as np
import pytest

from codecheck.codebert_score import CodeBERTScorer


def _stub_scorer(vectors: dict[str, list[float]]) -> CodeBERTScorer:
    """A scorer whose _embed returns preset (assumed L2-normalized) vectors, so the
    cosine/dissimilarity logic is tested without loading the model (or torch)."""
    scorer = CodeBERTScorer()
    scorer._embed = lambda codes: np.array([vectors[c] for c in codes], dtype=float)
    return scorer


def test_identical_embeddings_zero_dissimilarity():
    s = _stub_scorer({"main": [1.0, 0.0], "s": [1.0, 0.0]})
    score, per_sample = s.evaluate("main", ["s"])
    assert per_sample == [0.0]
    assert score == 0.0


def test_orthogonal_embeddings_max_dissimilarity():
    s = _stub_scorer({"main": [1.0, 0.0], "s": [0.0, 1.0]})
    score, per_sample = s.evaluate("main", ["s"])
    assert abs(per_sample[0] - 1.0) < 1e-6


def test_anti_parallel_clamped_to_one():
    s = _stub_scorer({"main": [1.0, 0.0], "s": [-1.0, 0.0]})  # cosine -1 -> dissim 2 -> clamp 1
    assert s.score("main", ["s"]) == 1.0


def test_mean_over_samples():
    s = _stub_scorer({"main": [1.0, 0.0], "a": [1.0, 0.0], "b": [0.0, 1.0]})
    score, per_sample = s.evaluate("main", ["a", "b"])
    assert per_sample[0] == 0.0
    assert abs(per_sample[1] - 1.0) < 1e-6
    assert abs(score - 0.5) < 1e-6


def test_empty_samples_scores_zero():
    s = _stub_scorer({"main": [1.0, 0.0]})
    assert s.score("main", []) == 0.0


def test_blank_main_is_max_divergence():
    s = CodeBERTScorer()  # returns early, never embeds
    assert s.evaluate("   \n", ["x", "y"]) == (1.0, [1.0, 1.0])


def test_blank_sample_is_max_divergence():
    s = _stub_scorer({"main": [1.0, 0.0], "": [0.0, 0.0]})
    score, per_sample = s.evaluate("main", [""])
    assert per_sample == [1.0]


def test_score_wraps_evaluate():
    s = _stub_scorer({"main": [1.0, 0.0]})
    assert s.score("main", []) == 0.0


@pytest.mark.skipif(os.environ.get("CODEBERT_INTEGRATION") != "1",
                    reason="set CODEBERT_INTEGRATION=1 to download + run the real model")
def test_real_model_identical_low_dissimilarity():
    s = CodeBERTScorer()
    code = "def f(x):\n    return x + 1\n"
    score, per_sample = s.evaluate(code, [code, "def g(y):\n    return y * 2 - 3\n"])
    assert per_sample[0] < 0.05            # identical code -> ~0
    assert all(0.0 <= d <= 1.0 for d in per_sample)
