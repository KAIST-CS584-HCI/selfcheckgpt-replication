from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt

from replication.entity import PassageInstance, PassageResponses, PassageResult, PassageScores
from replication.score.base import BaseScorer, REPO_ROOT


BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = "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"
MODEL = "qwen/qwen3.5-9b"
PROMPT_TEMPLATE = (
    "Context: {context}\n\n"
    "Sentence: {sentence}\n\n"
    "Is the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
)


class PromptScorer(BaseScorer):
    method_name = "prompt"
    default_dataset_path = REPO_ROOT / "data" / "dataset-original.json"

    def __init__(self, think: bool = False) -> None:
        self.think = think
        self.reasoning = "medium" if think else "none"
        self.max_tokens = 10000 if think else 5
        self.selfcheck = SelfCheckAPIPrompt(
            client_type="openai",
            base_url=BASE_URL,
            model=MODEL,
            api_key=API_KEY,
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
