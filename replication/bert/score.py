"""
SelfCheckGPT — BERTScore + NLI scoring for pre-generated data.

Loads one entry by index from a generated-text dataset JSON, scores each
sentence with SelfCheck-BERTScore and SelfCheck-NLI, and saves the result
as <index>.json in the same directory.

Usage:
    python3 -m replication.bert.score --index 0
    python3 -m replication.bert.score --index 42
"""

import argparse
import json
import os
import tempfile

import spacy

from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore, SelfCheckNLI
from replication.entity import GeneratedTextInstance

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset-generated-gpt-3.5-turbo-no-think.json')
RESULTS_DIR = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# NLI tokenizer compat shim (from selfcheck_replicate.py)
# ---------------------------------------------------------------------------

def _patch_nli_tokenizer(selfcheck_nli: SelfCheckNLI) -> SelfCheckNLI:
    """Newer transformers dropped `batch_encode_plus` — route it to __call__."""
    if not hasattr(selfcheck_nli.tokenizer, "batch_encode_plus"):
        def _compat(batch_text_or_text_pairs, **kwargs):
            return selfcheck_nli.tokenizer(batch_text_or_text_pairs, **kwargs)
        selfcheck_nli.tokenizer.batch_encode_plus = _compat
    return selfcheck_nli


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[GeneratedTextInstance]:
    with open(path) as f:
        return [GeneratedTextInstance.from_dict(item) for item in json.load(f)]


def save_result(result: dict, index: int) -> None:
    path = os.path.join(RESULTS_DIR, f"{index}.json")
    dir_ = os.path.dirname(path) or '.'
    with tempfile.NamedTemporaryFile('w', dir=dir_, delete=False, suffix='.tmp') as tmp:
        json.dump(result, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="Score one generated-text entry with BERTScore + NLI.")
_parser.add_argument("--index", type=int, required=True, help="index into the dataset JSON")


def main() -> None:
    args = _parser.parse_args()
    idx = args.index

    dataset = load_dataset(DATA_PATH)
    instance = dataset[idx]

    nlp = spacy.load("en_core_web_sm")

    selfcheck_bert = SelfCheckBERTScore()
    selfcheck_nli  = SelfCheckNLI()
    selfcheck_nli  = _patch_nli_tokenizer(selfcheck_nli)

    sentences = [sent.text.strip() for sent in nlp(instance.main_response).sents if sent.text.strip()]

    print(
        f"  Processing [example_id={instance.example_id}]: "
        f"{len(sentences)} sentences × {len(instance.sampled_passages)} samples ..."
    )

    bert_scores = selfcheck_bert.predict(
        sentences        = sentences,
        sampled_passages = instance.sampled_passages,
    )
    nli_scores = selfcheck_nli.predict(
        sentences        = sentences,
        sampled_passages = instance.sampled_passages,
    )

    result = {
        **instance.to_dict(),
        "sentences":   sentences,
        "bert_scores": bert_scores.tolist() if hasattr(bert_scores, "tolist") else list(bert_scores),
        "nli_scores":  nli_scores.tolist()  if hasattr(nli_scores,  "tolist") else list(nli_scores),
    }

    save_result(result, idx)
    print(f"    bert_scores: {[round(s, 3) for s in result['bert_scores']]}")
    print(f"    nli_scores:  {[round(s, 3) for s in result['nli_scores']]}")
    print(f"    saved → {os.path.join(RESULTS_DIR, f'{idx}.json')}")


if __name__ == '__main__':
    main()
