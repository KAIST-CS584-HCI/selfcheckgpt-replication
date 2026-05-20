"""
SelfCheckGPT — BERTScore scoring for pre-generated data.

Loads entries by index range from the generated-text dataset JSON, scores each
sentence with SelfCheck-BERTScore, and saves the result as bert/<index>.json.

Usage:
    python3 -m replication.score.bert --start 0 --end 119
    python3 -m replication.score.bert --start 0 --end 40
"""

import argparse
import json
import os
import tempfile

from tqdm import tqdm

from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
from replication.entity import PassageInstance

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dataset-generated.json')
RESULTS_DIR = os.path.join(os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[PassageInstance]:
    with open(path) as f:
        return [PassageInstance.from_dict(item) for item in json.load(f)]


def result_path(index: int) -> str:
    return os.path.join(RESULTS_DIR, f"{index}.json")


def save_result(result: dict, index: int) -> None:
    path = result_path(index)
    dir_ = os.path.dirname(path) or '.'
    os.makedirs(dir_, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=dir_, delete=False, suffix='.tmp') as tmp:
        json.dump(result, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="Score generated-text entries with SelfCheck-BERTScore.")
_parser.add_argument("--start", type=int, required=True, help="start index (inclusive)")
_parser.add_argument("--end",   type=int, required=True, help="end index (exclusive)")

def main() -> None:
    args = _parser.parse_args()

    indices = list(range(args.start, args.end))
    if not indices:
        print("Nothing to do.")
        return

    dataset = load_dataset(DATA_PATH)

    print("Loading SelfCheck-BERTScore ...")
    selfcheck_bert = SelfCheckBERTScore()

    for idx in tqdm(indices, desc="BERTScore scoring"):
        instance = dataset[idx]

        bert_scores = selfcheck_bert.predict(
            sentences        = instance.main_sentences,
            sampled_passages = instance.sample_passages,
        )

        result = {
            "dataset_idx":       idx,
            "wiki_bio_test_idx": instance.wiki_bio_test_idx,
            "wiki_bio_text":     instance.wiki_bio_text,
            "main_passage":      instance.main_passage,
            "main_sentences":    instance.main_sentences,
            "annotation":        instance.annotation,
            "sample_passages":   instance.sample_passages,
            "scores": {
                "bert": bert_scores.tolist() if hasattr(bert_scores, "tolist") else list(bert_scores),
            },
            "responses": {},
        }

        save_result(result, idx)

    print(f"\nDone: {len(indices)} entries.")


if __name__ == '__main__':
    main()
