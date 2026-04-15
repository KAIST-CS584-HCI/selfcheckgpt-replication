from dataclasses import dataclass


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
