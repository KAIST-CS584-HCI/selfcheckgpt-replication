from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1]
RESULTS_PATH = RESULTS_DIR / "results.json"

LABEL_SCORE = {
    "accurate": 0.0,
    "minor_inaccurate": 0.5,
    "major_inaccurate": 1.0,
}

VARIANTS = ["prompt", "bert", "nli"]

PAPER_BASELINES = {
    "bert": {"nonfact": 81.96, "nonfact_star": 45.96, "factual": 44.23, "pearson": 58.18, "spearman": 55.90},
    "nli": {"nonfact": 92.50, "nonfact_star": 45.17, "factual": 66.08, "pearson": 74.14, "spearman": 73.78},
    "prompt": {"nonfact": 93.42, "nonfact_star": 53.19, "factual": 67.09, "pearson": 78.32, "spearman": 78.30},
}

PAPER_LABELS = {
    "bert": "SelfCk-BERTScore (paper, ChatGPT)",
    "nli": "SelfCk-NLI       (paper, ChatGPT)",
    "prompt": "SelfCk-Prompt    (paper, ChatGPT)",
}

VARIANT_LABELS = {
    "bert": "SelfCk-BERTScore",
    "nli": "SelfCk-NLI      ",
    "prompt": "SelfCk-Prompt   ",
}
