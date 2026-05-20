import os
from dataclasses import dataclass

from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt

from replication.entity import PassageInstance, PassageResponses, PassageResult, PassageScores
from replication.score.base import BaseScorer, REPO_ROOT


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3.5-9b"
PROMPT_TEMPLATE = (
    "Context: {context}\n\n"
    "Sentence: {sentence}\n\n"
    "Is the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
)


@dataclass(frozen=True)
class PromptScorerConfig:
    api_key: str
    model: str = DEFAULT_OPENROUTER_MODEL
    base_url: str = DEFAULT_OPENROUTER_BASE_URL


def get_openrouter_config() -> PromptScorerConfig:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY. Create .env from .env.example or set it in the environment."
        )

    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip() or DEFAULT_OPENROUTER_MODEL
    base_url = os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).strip()
    base_url = base_url or DEFAULT_OPENROUTER_BASE_URL
    return PromptScorerConfig(api_key=api_key, model=model, base_url=base_url)


class PromptScorer(BaseScorer):
    method_name = "prompt"
    default_dataset_path = REPO_ROOT / "data" / "dataset-original.json"

    def __init__(self, think: bool = False) -> None:
        self.think = think
        self.reasoning = "medium" if think else "none"
        self.max_tokens = 10000 if think else 5
        config = get_openrouter_config()
        self.selfcheck = SelfCheckAPIPrompt(
            client_type="openai",
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
        )
        self.selfcheck.set_prompt_template(PROMPT_TEMPLATE)

    def score(self, dataset_idx: int, passage: PassageInstance) -> PassageResult:
        print(
            f"  Processing [wiki_bio_test_idx={passage.wiki_bio_test_idx}]: "
            f"{len(passage.main_sentences)} sentences × {len(passage.sample_passages)} samples ..."
        )
        scores, raw_responses = self.selfcheck.predict(
            sentences=passage.main_sentences,
            sampled_passages=passage.sample_passages,
            verbose=True,
            max_tokens=self.max_tokens,
            reasoning=self.reasoning,
        )
        return PassageResult(
            dataset_idx=dataset_idx,
            wiki_bio_test_idx=passage.wiki_bio_test_idx,
            wiki_bio_text=passage.wiki_bio_text,
            main_passage=passage.main_passage,
            main_sentences=passage.main_sentences,
            annotation=passage.annotation,
            sample_passages=passage.sample_passages,
            scores=PassageScores(prompt=scores.tolist()),
            responses=PassageResponses(prompt=raw_responses),
        )
