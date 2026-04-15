"""
SelfCheckGPT Prompt Method Replication
---------------------------------------
Runs SelfCheckAPIPrompt over a range of WikiBio passages.
Each result is saved as <idx>.json in the same directory.

Run:
    python3 -m replication.prompt.main --start <start_idx> --end <end_idx>

Example:
    python3 -m replication.prompt.main --start 119 --end 120
    → processes dataset indices 119 and 120, saves 119.json and 120.json
"""
import argparse
import os
import json
import tempfile
import time
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
from replication.entity import PassageInput, PassageResult

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL         = "https://ollama.makinteract.com/v1/"
API_KEY          = "haha"
MODEL            = "qwen3.5:9b-q8_0"
DATA_PATH        = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.json')
RESULTS_DIR      = os.path.dirname(__file__)
PROMPT_TEMPLATE  = (
    "Context: {context}\n\n"
    "Sentence: {sentence}\n\n"
    "Is the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
)

MAX_TOKENS       = 5  # only "Yes" or "No" needed; thinking is disabled via extra_body
REQUEST_TIMEOUT  = 200   # seconds per API call before raising an error
MAX_RETRIES      = 3     # retry attempts per passage on transient errors
RETRY_DELAY      = 10    # seconds to wait between retries

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[PassageInput]:
    with open(path) as f:
        return [PassageInput.from_dict(item) for item in json.load(f)]


RESULTS_PATH = os.path.join(RESULTS_DIR, "results.json")


def load_done_set() -> set[int]:
    if not os.path.exists(RESULTS_PATH):
        return set()
    with open(RESULTS_PATH) as f:
        return {r["wiki_bio_test_idx"] for r in json.load(f)}


def save_result(result: PassageResult, idx: int) -> None:
    path = os.path.join(RESULTS_DIR, f"{idx}.json")
    dir_ = os.path.dirname(path) or '.'
    with tempfile.NamedTemporaryFile('w', dir=dir_, delete=False, suffix='.tmp') as tmp:
        json.dump(result.to_dict(), tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="Run SelfCheckAPIPrompt over a range of WikiBio passages.")
_parser.add_argument("--index", type=int, required=True, help="index of the passage to process (wiki_bio_test_idx)")

def main() -> None:
    args = _parser.parse_args()

    idx = args.index

    dataset  = load_dataset(DATA_PATH)
    passage = dataset[idx]
    done     = load_done_set()

    checker = SelfCheckAPIPrompt(
        client_type="openai",
        base_url=BASE_URL,
        model=MODEL,
        api_key=API_KEY,
        timeout=REQUEST_TIMEOUT,
    )
    checker.set_prompt_template(PROMPT_TEMPLATE)

    wiki_idx = passage.wiki_bio_test_idx
    if wiki_idx in done:
        print(f"  Skipping [wiki_bio_test_idx={wiki_idx} (already in results.json)")
        return

    print(
        f"  Processing [wiki_bio_test_idx={wiki_idx}]: "
        f"{len(passage.sentences)} sentences × {len(passage.sampled_passages)} samples ..."
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            sent_scores = checker.predict(
                sentences        = passage.sentences,
                sampled_passages = passage.sampled_passages,
                verbose          = True,
            )
            break
        except Exception as exc:
            print(f"    attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                print(f"    retrying in {RETRY_DELAY}s ...")
                time.sleep(RETRY_DELAY)

    result = PassageResult(
        wiki_bio_test_idx = wiki_idx,
        sent_scores       = sent_scores.tolist(),
        annotation        = passage.annotation,
    )
    save_result(result, idx)
    print(f"    scores: {[round(s, 3) for s in result.sent_scores]}")
    print(f"    saved → {os.path.join(RESULTS_DIR, f'{idx}.json')}")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
