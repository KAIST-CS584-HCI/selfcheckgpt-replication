"""
SelfCheckGPT — BERTScore scoring for pre-generated data.

Loads entries by index range from the generated-text dataset JSON, scores each
sentence with SelfCheck-BERTScore, and saves the result as bert/<index>.json.

Usage:
    python3 -m replication.bert.score_bert --start 0 --end 119
    python3 -m replication.bert.score_bert --start 0 --end 40 --skip-existing
"""

import argparse
import json
import os
import tempfile

import spacy
from tqdm import tqdm

from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
from replication.entity import GeneratedTextInstance

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset-generated-gpt-3.5-turbo-no-think.json')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'bert')


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[GeneratedTextInstance]:
    with open(path) as f:
        return [GeneratedTextInstance.from_dict(item) for item in json.load(f)]


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

    print("Loading spaCy + SelfCheck-BERTScore ...")
    nlp = spacy.load("en_core_web_sm")
    selfcheck_bert = SelfCheckBERTScore()

    for idx in tqdm(indices, desc="BERTScore scoring"):
        instance = dataset[idx]
        sentences = [sent.text.strip() for sent in nlp(instance.main_response).sents if sent.text.strip()]

        bert_scores = selfcheck_bert.predict(
            sentences        = sentences,
            sampled_passages = instance.sampled_passages,
        )

        result = {
            **instance.to_dict(),
            "sentences":   sentences,
            "bert_scores": bert_scores.tolist() if hasattr(bert_scores, "tolist") else list(bert_scores),
        }

        save_result(result, idx)

    print(f"\nDone: {len(indices)} entries.")


if __name__ == '__main__':
    main()
