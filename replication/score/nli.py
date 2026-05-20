import torch

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

from replication.entity import PassageInstance, PassageResponses, PassageResult, PassageScores
from replication.score.base import BaseScorer, REPO_ROOT


def _patch_nli_tokenizer(selfcheck_nli: SelfCheckNLI) -> SelfCheckNLI:
    """Newer transformers dropped `batch_encode_plus` — route it to __call__."""
    if not hasattr(selfcheck_nli.tokenizer, "batch_encode_plus"):
        def _compat(batch_text_or_text_pairs, **kwargs):
            return selfcheck_nli.tokenizer(batch_text_or_text_pairs, **kwargs)
        selfcheck_nli.tokenizer.batch_encode_plus = _compat
    return selfcheck_nli


class NliScorer(BaseScorer):
    method_name = "nli"
    default_dataset_path = REPO_ROOT / "data" / "dataset-generated.json"

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SelfCheck-NLI on {self.device} ...")
        self.selfcheck = _patch_nli_tokenizer(SelfCheckNLI(device=self.device))

    def score(self, dataset_idx: int, passage: PassageInstance) -> PassageResult:
        scores = self.selfcheck.predict(
            sentences=passage.main_sentences,
            sampled_passages=passage.sample_passages,
        )
        return PassageResult(
            dataset_idx=dataset_idx,
            wiki_bio_test_idx=passage.wiki_bio_test_idx,
            wiki_bio_text=passage.wiki_bio_text,
            main_passage=passage.main_passage,
            main_sentences=passage.main_sentences,
            annotation=passage.annotation,
            sample_passages=passage.sample_passages,
            scores=PassageScores(nli=scores.tolist() if hasattr(scores, "tolist") else list(scores)),
            responses=PassageResponses(),
        )
