"""
SelfCheckGPT — NLI scoring for pre-generated data.

Loads entries by index range from the generated-text dataset JSON, scores each
sentence with SelfCheck-NLI, and saves the result as nli_<index>.json.

Usage:
    python3 -m replication.bert.score_nli --start 0 --end 119
    python3 -m replication.bert.score_nli --start 0 --end 40 --skip-existing
"""

import argparse
import json
import os
import tempfile

import spacy
import torch
from tqdm import tqdm

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
from replication.entity import GeneratedTextInstance

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset-generated-gpt-3.5-turbo-no-think.json')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'nli')


# ---------------------------------------------------------------------------
# NLI tokenizer compat shim
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


def result_path(index: int) -> str:
    return os.path.join(RESULTS_DIR, f"{index}.json")


def save_result(result: dict, index: int) -> None:
    path = result_path(index)
    dir_ = os.path.dirname(path) or '.'
    with tempfile.NamedTemporaryFile('w', dir=dir_, delete=False, suffix='.tmp') as tmp:
        json.dump(result, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="Score generated-text entries with SelfCheck-NLI.")
_parser.add_argument("--start", type=int, required=True, help="start index (inclusive)")
_parser.add_argument("--end",   type=int, required=True, help="end index (exclusive)")

def main() -> None:
    args = _parser.parse_args()

    indices = list(range(args.start, args.end))
    if not indices:
        print("Nothing to do.")
        return

    dataset = load_dataset(DATA_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading spaCy + SelfCheck-NLI on {device} ...")
    
    nlp = spacy.load("en_core_web_sm")
    selfcheck_nli = SelfCheckNLI(device=device)
    selfcheck_nli = _patch_nli_tokenizer(selfcheck_nli)

    for idx in tqdm(indices, desc="NLI scoring"):
        instance = dataset[idx]
        sentences = [sent.text.strip() for sent in nlp(instance.main_response).sents if sent.text.strip()]

        print(
            f"  Processing [#{idx} example_id={instance.example_id}]: "
            f"{len(sentences)} sentences × {len(instance.sampled_passages)} samples ..."
        )

        nli_scores = selfcheck_nli.predict(
            sentences        = sentences,
            sampled_passages = instance.sampled_passages,
        )

        result = {
            **instance.to_dict(),
            "sentences":  sentences,
            "nli_scores": nli_scores.tolist() if hasattr(nli_scores, "tolist") else list(nli_scores),
        }

        save_result(result, idx)
        print(f"    nli_scores: {[round(s, 3) for s in result['nli_scores']]}")
        print(f"    saved → {result_path(idx)}")

    print(f"\nDone: {len(indices)} entries.")


if __name__ == '__main__':
    main()
