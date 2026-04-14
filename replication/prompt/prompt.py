"""
SelfCheckGPT Prompt Method Replication
---------------------------------------
Runs SelfCheckAPIPrompt over all 238 WikiBio passages using a local Ollama endpoint.
Results are saved to results.json (one entry per passage) for later metrics evaluation.

Run:
    python prompt.py
"""
import sys
import os
import json
import tempfile
from dataclasses import dataclass

# Make the selfcheckgpt package importable without installing it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "https://ollama.makinteract.com/v1/"
MODEL           = "qwen3.5:9b-q8_0"
DATA_PATH       = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.json')
RESULTS_PATH    = os.path.join(os.path.dirname(__file__), 'results.json')

# Prepend /no_think to suppress Qwen3.5's chain-of-thought reasoning.
# Without this, the model emits <think>...</think> tokens before "Yes"/"No",
# causing text_postprocessing() to classify the response as "n/a" (score 0.5).
PROMPT_TEMPLATE = (
    "/no_think\n\n"
    "Context: {context}\n\n"
    "Sentence: {sentence}\n\n"
    "Is the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PassageInput:
    """One passage from dataset.json, with only the fields needed for prediction."""
    wiki_bio_test_idx: int
    sentences:         list[str]   # gpt3_sentences — the text to be evaluated
    sampled_passages:  list[str]   # gpt3_text_samples — 20 stochastic samples
    annotation:        list[str]   # human labels per sentence, carried to output

    @classmethod
    def from_dict(cls, item: dict) -> "PassageInput":
        return cls(
            wiki_bio_test_idx = item['wiki_bio_test_idx'],
            sentences         = item['gpt3_sentences'],
            sampled_passages  = item['gpt3_text_samples'],
            annotation        = item['annotation'],
        )


@dataclass
class PassageResult:
    """Prediction output for one passage, ready to be serialised to results.json."""
    wiki_bio_test_idx: int
    sent_scores:       list[float]  # one per sentence; 0.0 = factual, 1.0 = hallucinated
    annotation:        list[str]    # carried from PassageInput for later eval

    def to_dict(self) -> dict:
        return {
            'wiki_bio_test_idx': self.wiki_bio_test_idx,
            'sent_scores':       self.sent_scores,
            'annotation':        self.annotation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PassageResult":
        return cls(
            wiki_bio_test_idx = d['wiki_bio_test_idx'],
            sent_scores       = d['sent_scores'],
            annotation        = d['annotation'],
        )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[PassageInput]:
    with open(path) as f:
        return [PassageInput.from_dict(item) for item in json.load(f)]


def load_results(path: str) -> list[PassageResult]:
    """Load existing results for checkpoint/resume. Returns [] if file absent."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        try:
            return [PassageResult.from_dict(d) for d in json.load(f)]
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Checkpoint file {path} is corrupted: {exc}") from exc


def save_results(path: str, results: list[PassageResult]) -> None:
    dir_ = os.path.dirname(path) or '.'
    with tempfile.NamedTemporaryFile('w', dir=dir_, delete=False, suffix='.tmp') as tmp:
        json.dump([r.to_dict() for r in results], tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset = load_dataset(DATA_PATH)
    results = load_results(RESULTS_PATH)
    done_ids = {r.wiki_bio_test_idx for r in results}

    print(f"Dataset: {len(dataset)} passages | Already done: {len(done_ids)}")

    checker = SelfCheckAPIPrompt(
        client_type="openai",
        base_url=OLLAMA_BASE_URL,
        model=MODEL,
    )
    checker.set_prompt_template(PROMPT_TEMPLATE)

    for i, passage in enumerate(dataset):
        if passage.wiki_bio_test_idx in done_ids:
            print(f"[{i+1}/{len(dataset)}] Skipping {passage.wiki_bio_test_idx} (already done)")
            continue

        print(
            f"[{i+1}/{len(dataset)}] Passage {passage.wiki_bio_test_idx}: "
            f"{len(passage.sentences)} sentences × {len(passage.sampled_passages)} samples ..."
        )

        try:
            sent_scores = checker.predict(
                sentences        = passage.sentences,
                sampled_passages = passage.sampled_passages,
                verbose          = True,
            )
        except Exception as exc:
            print(f"  ERROR on passage {passage.wiki_bio_test_idx}: {exc} — skipping")
            continue

        result = PassageResult(
            wiki_bio_test_idx = passage.wiki_bio_test_idx,
            sent_scores       = sent_scores.tolist(),
            annotation        = passage.annotation,
        )
        results.append(result)
        save_results(RESULTS_PATH, results)
        print(f"  scores: {[round(s, 3) for s in result.sent_scores]}")

    print(f"\nDone. {len(results)} passages saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
