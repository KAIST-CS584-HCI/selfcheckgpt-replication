from dataclasses import dataclass, field


@dataclass
class PassagePromptResult:
    """Prediction output for one passage, ready to be serialised to results.json."""
    dataset_idx:       int
    wiki_bio_test_idx: int
    main_passage:      str
    sample_passages:   list[str]
    sentences:         list[str]
    annotation:       list[str]                 # carried from PassageInput for later eval
    sentence_scores:   list[float]               # one per sentence; 0.0 = factual, 1.0 = hallucinated
    raw_responses:     list[list[str | None]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'dataset_idx':       self.dataset_idx,
            'wiki_bio_test_idx': self.wiki_bio_test_idx,
            'main_passage':      self.main_passage,
            'sample_passages':   self.sample_passages,
            'sentences':         self.sentences,
            'annotation':       self.annotation,
            'sentence_scores':   self.sentence_scores,
            'raw_responses':     self.raw_responses,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PassagePromptResult":
        return cls(
            dataset_idx       = d['dataset_idx'],
            wiki_bio_test_idx = d['wiki_bio_test_idx'],
            main_passage      = d['main_passage'],
            sample_passages   = d['sample_passages'],
            sentences         = d['sentences'],
            annotation       = d['annotation'],
            sentence_scores   = d['sentence_scores'],
            raw_responses     = d.get('raw_responses', []),
        )
