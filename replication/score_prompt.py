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
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
from replication.entity import PassageInstance, PassageResult

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# BASE_URL         = "https://openrouter.ai/api/v1"
# API_KEY          = "your API"
# MODEL            = "google/gemma-4-31b-it"
BASE_URL         = "https://ollama.makinteract.com/v1/"
API_KEY          = "haha"
MODEL            = "qwen3.5:9b-q8_0"
DATA_PATH        = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.json')
RESULTS_DIR      = os.path.dirname(__file__)
RESULTS_PATH     = os.path.join(RESULTS_DIR, "results.json")
PROMPT_TEMPLATE  = (
    "Context: {context}\n\n"
    "Sentence: {sentence}\n\n"
    "Is the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[PassageInstance]:
    with open(path) as f:
        return [PassageInstance.from_dict(item) for item in json.load(f)]

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
_parser.add_argument("--think", action="store_true", default=False, help="enable reasoning effort in API calls")

def main() -> None:
    args = _parser.parse_args()

    idx = args.index
    think = "medium" if args.think else "none"
    max_token = 10000 if args.think else 5

    dataset  = load_dataset(DATA_PATH)
    passage = dataset[idx]

    checker = SelfCheckAPIPrompt(
        client_type="openai",
        base_url=BASE_URL,
        model=MODEL,
        api_key=API_KEY,
    )
    checker.set_prompt_template(PROMPT_TEMPLATE)

    wiki_idx = passage.wiki_bio_test_idx
    print(
        f"  Processing [wiki_bio_test_idx={wiki_idx}]: "
        f"{len(passage.sentences)} sentences × {len(passage.sampled_passages)} samples ..."
    )

    try:
        sent_scores, raw_responses = checker.predict(
            sentences        = passage.sentences,
            sampled_passages = passage.sampled_passages,
            verbose          = True,
            max_tokens       = max_token,
            reasoning        = think,
        )

    except Exception as exc:
        print(f"  Error processing index {idx} (wiki_bio_test_idx={wiki_idx}): {exc}")
        return

    result = PassageResult(
        dataset_idx       = idx,
        wiki_bio_test_idx = wiki_idx,
        gpt3_text         = passage.gpt3_text,
        sentences         = passage.sentences,
        sentence_scores   = sent_scores.tolist(),
        annotations       = passage.annotation,
        raw_responses     = raw_responses,
    )

    save_result(result, idx)
    print(f"    scores: {[round(s, 3) for s in result.sentence_scores]}")
    print(f"    saved → {os.path.join(RESULTS_DIR, f'{idx}.json')}")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
