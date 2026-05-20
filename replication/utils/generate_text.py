"""
Generate text and stochastic samples for a single passage in the dataset.

Output structure mirrors potsawee/wiki_bio_gpt3_hallucination:
  gpt3_text, wiki_bio_text, gpt3_sentences, annotation,
  wiki_bio_test_idx, gpt3_text_samples

Usage
-----
    export OPENROUTER_API_KEY=sk-or-...
    python3 generate_text.py --index 0
    python3 generate_text.py --index 5 --samples 10 --output-dir generated/
"""

import argparse
import json
import os
import time

import spacy
from openai import OpenAI

from replication.entity import PassageInstance


MODEL_NAME = "openai/gpt-3.5-turbo"
BASE_URL = "https://openrouter.ai/api/v1"
DATASET_PATH = "data/dataset.json"


def extract_concept(wiki_text: str) -> str:
    return wiki_text.split(",")[0]


def make_generation_prompt(concept: str) -> str:
    return f"This is a Wikipedia passage about {concept}:"


def generate_text(
    client: OpenAI,
    prompt: str,
    think: str = "none",
    temperature: float = 1.0,
    retries: int = 3,
) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                reasoning_effort=think,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Retry {attempt + 1}: {e}")
            time.sleep(2)
    raise RuntimeError("Generation failed after retries")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text and samples for one passage")
    parser.add_argument("--index", type=int, required=True,
                        help="dataset ordinal index of the passage to process")
    parser.add_argument("--samples", type=int, default=10,
                        help="number of stochastic samples to generate (default: 10)")
    parser.add_argument("--output-dir", type=str, default="replication/generated",
                        help="directory for output JSON files (default: replication/generated/)")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENROUTER_API_KEY", ""),
                        help="OpenRouter API key (falls back to OPENROUTER_API_KEY env var)")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("No API key. Set OPENROUTER_API_KEY env var or pass --api-key.")

    out_path = os.path.join(args.output_dir, f"{args.index}.json")
    if os.path.exists(out_path):
        print(f"Already exists: {out_path}")
        return

    client = OpenAI(api_key=args.api_key, base_url=BASE_URL)
    os.makedirs(args.output_dir, exist_ok=True)

    nlp = spacy.load("en_core_web_sm")

    with open(DATASET_PATH) as f:
        dataset = [PassageInstance.from_dict(item) for item in json.load(f)]
    ex = dataset[args.index]

    concept = extract_concept(ex.wiki_bio_text)
    prompt = make_generation_prompt(concept)

    print("Generating main passage (T=0.0)...")
    gpt3_text = generate_text(client, prompt, temperature=0.0)
    gpt3_sentences = [s.text.strip() for s in nlp(gpt3_text).sents if s.text.strip()]

    gpt3_text_samples = []
    for i in range(args.samples):
        text = generate_text(client, prompt, temperature=1.0)
        gpt3_text_samples.append(text)
        print(f"  sample {i + 1}/{args.samples} done")

    result = {
        "gpt3_text": gpt3_text,
        "wiki_bio_text": ex.wiki_bio_text,
        "gpt3_sentences": gpt3_sentences,
        "annotation": ex.annotation,
        "wiki_bio_test_idx": ex.wiki_bio_test_idx,
        "gpt3_text_samples": gpt3_text_samples,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
