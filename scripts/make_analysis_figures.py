"""Generate the Improvement-2 analysis figures (MBPP+, both models) into
docs/analysis/images/. Reuses the exact label/score construction and trapezoidal AUC-PR from
codecheck.evaluate, so every figure's annotated AUC matches `run_codecheck.py evaluate`.

Run from the repo root:  python scripts/make_analysis_figures.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import codecheck from repo root

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve

from codecheck.pipeline import load_results
from codecheck.evaluate import (
    auc_pr_detect_incorrect, auc_pr_detect_correct, prevalence_baseline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO_ROOT / "docs" / "analysis" / "images"

# (label, results file). MBPP+ 300 on the two local models.
MODELS = [
    ("gemma4-31B", REPO_ROOT / "results" / "codecheck-mbpp-300-gemma4-31B.json"),
    ("qwen3.5-9B", REPO_ROOT / "results" / "codecheck-mbpp-300-qwen3.5-9B.json"),
]
METHODS = ["exec", "prompt", "ast", "code_bert"]
METHOD_COLORS = {"exec": "#2a9d8f", "prompt": "#4a7fb5", "ast": "#e9a23b", "code_bert": "#c1432e"}


# --- low-level: curve + score extraction (mirrors codecheck.evaluate) ---

def _pr_points(results, method: str, detect: str):
    """(recall, precision) for one method. detect='incorrect' uses the score as-is with
    incorrect=positive; detect='correct' negates the score with correct=positive."""
    if detect == "incorrect":
        y_true = [0 if r.is_correct else 1 for r in results]
        scores = [r.scores[method] for r in results]
    else:
        y_true = [1 if r.is_correct else 0 for r in results]
        scores = [-r.scores[method] for r in results]
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return recall, precision


def _method_scores_by_class(results, method: str):
    """(correct_scores, incorrect_scores) for one method's histogram."""
    correct = [r.scores[method] for r in results if r.is_correct]
    incorrect = [r.scores[method] for r in results if not r.is_correct]
    return correct, incorrect


# --- figures ---

def plot_auc_bars(model_data) -> Path:
    """Headline: grouped AUC-PR bars (detect-incorrect vs detect-correct) per method, one
    facet per model, with the no-skill prevalence baseline drawn for detect-incorrect."""
    fig, axes = plt.subplots(1, len(model_data), figsize=(11, 4.5), sharey=True)
    x = range(len(METHODS))
    width = 0.38
    for ax, (label, results) in zip(axes, model_data):
        inc = [auc_pr_detect_incorrect(results, m) for m in METHODS]
        cor = [auc_pr_detect_correct(results, m) for m in METHODS]
        base = prevalence_baseline(results)
        ax.bar([i - width / 2 for i in x], inc, width, label="detect-incorrect", color="#c1432e")
        ax.bar([i + width / 2 for i in x], cor, width, label="detect-correct", color="#2a9d8f")
        ax.axhline(base, ls="--", c="#555", lw=1, label=f"baseline (prevalence {base:.2f})")
        ax.set_xticks(list(x))
        ax.set_xticklabels(METHODS, rotation=20)
        ax.set_title(label)
        ax.set_ylim(0, 1)
        for i, v in enumerate(inc):
            ax.text(i - width / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("AUC-PR")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("AUC-PR by method (MBPP+, n=300) — detect-incorrect vs detect-correct")
    fig.tight_layout()
    return _save(fig, "fig_auc_bars.png")


def plot_pr_curves(label: str, results) -> Path:
    """Per model: two panels (detect-incorrect, detect-correct), one curve per method, with the
    random baseline and the trapezoidal AUC annotated in the legend."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    panels = [("incorrect", "detect-incorrect", prevalence_baseline(results)),
              ("correct", "detect-correct", 1 - prevalence_baseline(results))]
    for ax, (detect, title, base) in zip(axes, panels):
        for m in METHODS:
            recall, precision = _pr_points(results, m, detect)
            a = (auc_pr_detect_incorrect if detect == "incorrect" else auc_pr_detect_correct)(results, m)
            ax.plot(recall, precision, color=METHOD_COLORS[m], lw=1.5, label=f"{m} ({a:.3f})")
        ax.axhline(base, ls="--", c="#888", lw=1, label=f"random ({base:.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Precision-Recall curves — {label} (MBPP+, n=300)")
    fig.tight_layout()
    return _save(fig, f"fig_pr_curves_{label}.png")


def plot_histograms(label: str, results) -> Path:
    """Per model: per-method stacked score histograms (correct vs incorrect) over 10 bins,
    exposing the tie pile-up near 0 that makes the scalar AUC fragile."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    bins = [i / 10 for i in range(11)]
    for ax, m in zip(axes.ravel(), METHODS):
        correct, incorrect = _method_scores_by_class(results, m)
        ax.hist([correct, incorrect], bins=bins, stacked=True,
                color=["#2a9d8f", "#c1432e"], label=["correct", "incorrect"])
        ax.set_title(m)
        ax.set_xlabel("inconsistency score")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle(f"Score distribution by class — {label} (MBPP+, n=300)")
    fig.tight_layout()
    return _save(fig, f"fig_hist_{label}.png")


def _save(fig, name: str) -> Path:
    path = IMAGES_DIR / name
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    model_data = [(label, load_results(path)) for label, path in MODELS]
    written = [plot_auc_bars(model_data)]
    for label, results in model_data:
        written.append(plot_pr_curves(label, results))
        written.append(plot_histograms(label, results))
    for p in written:
        print(f"wrote {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
