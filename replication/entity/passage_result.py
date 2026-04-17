from dataclasses import dataclass, field


@dataclass
class PassageResult:
    """Prediction output for one passage, ready to be serialised to results.json."""
    dataset_idx:       int
    wiki_bio_test_idx: int
    gpt3_text:         str
    sentences:         list[str]
    sentence_scores:   list[float]               # one per sentence; 0.0 = factual, 1.0 = hallucinated
    annotations:       list[str]                 # carried from PassageInput for later eval
    raw_responses:     list[list[str | None]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'dataset_idx':       self.dataset_idx,
            'wiki_bio_test_idx': self.wiki_bio_test_idx,
            'gpt3_text':         self.gpt3_text,
            'sentences':         self.sentences,
            'sentence_scores':   self.sentence_scores,
            'annotations':       self.annotations,
            'raw_responses':     self.raw_responses,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PassageResult":
        return cls(
            dataset_idx       = d['dataset_idx'],
            wiki_bio_test_idx = d['wiki_bio_test_idx'],
            gpt3_text         = d['gpt3_text'],
            sentences         = d['sentences'],
            sentence_scores   = d['sentence_scores'],
            annotations       = d['annotations'],
            raw_responses     = d.get('raw_responses', []),
        )
