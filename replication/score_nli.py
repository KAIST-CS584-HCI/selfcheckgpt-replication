"""
SelfCheckGPT — NLI scoring for wiki_bio dataset.

Loads entries by index range from dataset-generated-samples-gpt-3.5-turbo.json
using PassageGeneratedInstance, scores each sentence with SelfCheck-NLI, and
saves the result as <index>.json.

Usage:
    python3 -m replication.score_nli --start 0 --end 119
    python3 -m replication.score_nli --start 0 --end 119 --skip-existing
"""

import argparse
import json
import os
import tempfile

import torch
from tqdm import tqdm

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
from replication.entity import PassageGeneratedInstance

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH   = os.path.join(os.path.dirname(__file__), 'data', 'dataset-generated-samples-gpt-3.5-turbo.json')
RESULTS_DIR = os.path.join(os.path.dirname(__file__))


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

def load_dataset(path: str) -> list[PassageGeneratedInstance]:
    with open(path) as f:
        return [PassageGeneratedInstance.from_dict(item) for item in json.load(f)]


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

_parser = argparse.ArgumentParser(description="Score wiki_bio dataset entries with SelfCheck-NLI.")
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
    print(f"Loading SelfCheck-NLI on {device} ...")
    selfcheck_nli = _patch_nli_tokenizer(SelfCheckNLI(device=device))

    for idx in tqdm(indices, desc="NLI scoring"):
        instance = dataset[idx]

        nli_scores = selfcheck_nli.predict(
            sentences        = instance.main_sentences,
            sampled_passages = instance.sample_passages,
        )

        result = {
            "dataset_idx":       idx,
            "wiki_bio_test_idx": instance.wiki_bio_test_idx,
            "main_passage":      instance.main_passage,
            "main_sentences":    instance.main_sentences,
            "annotation":        instance.annotation,
            "sample_passages":   instance.sample_passages,
            "wiki_bio_text":     instance.wiki_bio_text,
            "nli_scores":        nli_scores.tolist() if hasattr(nli_scores, "tolist") else list(nli_scores),
        }

        save_result(result, idx)

    print(f"\nDone: {len(indices)} entries.")


if __name__ == '__main__':
    main()
