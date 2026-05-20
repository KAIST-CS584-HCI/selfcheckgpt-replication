from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore

from replication.entity import PassageInstance, PassageResponses, PassageResult, PassageScores
from replication.score.base import BaseScorer, REPO_ROOT


class BertScorer(BaseScorer):
    method_name = "bert"
    default_dataset_path = REPO_ROOT / "data" / "dataset-generated.json"

    def __init__(self) -> None:
        print("Loading SelfCheck-BERTScore ...")
        self.selfcheck = SelfCheckBERTScore()

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
            scores=PassageScores(bert=scores.tolist() if hasattr(scores, "tolist") else list(scores)),
            responses=PassageResponses(),
        )
