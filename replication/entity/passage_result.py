from dataclasses import dataclass, field


@dataclass
class PassageResult:
    """Prediction output for one passage, ready to be serialised to results.json."""
    wiki_bio_test_idx: int
    sentence_scores:       list[float]            # one per sentence; 0.0 = factual, 1.0 = hallucinated
    annotations:        list[str]              # carried from PassageInput for later eval
    raw_responses:     list[list[str | None]] = field(default_factory=list)
    # raw_responses[sent_i][sample_i] = raw API message string ("Yes"/"No"/None)

    def to_dict(self) -> dict:
        return {
            'wiki_bio_test_idx': self.wiki_bio_test_idx,
            'sentence_scores':       self.sentence_scores,
            'annotations':        self.annotations,
            'raw_responses':     self.raw_responses,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PassageResult":
        return cls(
            wiki_bio_test_idx = d['wiki_bio_test_idx'],
            sentence_scores       = d['sentence_scores'],
            annotations        = d['annotations'],
            raw_responses     = d.get('raw_responses', []),
        )
