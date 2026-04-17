from dataclasses import dataclass, field


@dataclass
class PromptInstance:
    """One result entry from a prompt-based selfcheck run (e.g. result-prompt-*.json)."""
    dataset_idx:      int
    wiki_bio_test_idx: int
    sentences:        list[str]
    annotations:      list[str]
    score_matrix:     list[list[float]]          # [sent_i][sample_i] — 0.0/1.0 per raw response
    sentence_scores:  list[float]                # mean score per sentence
    passage_score:    float
    n_undefined:      int
    raw_responses:    list[list[str | None]] = field(default_factory=list)
    # raw_responses[sent_i][sample_i] = "Yes" / "No" / None

    def to_dict(self) -> dict:
        return {
            'dataset_idx':       self.dataset_idx,
            'wiki_bio_test_idx': self.wiki_bio_test_idx,
            'sentences':         self.sentences,
            'annotations':       self.annotations,
            'score_matrix':      self.score_matrix,
            'sentence_scores':   self.sentence_scores,
            'passage_score':     self.passage_score,
            'n_undefined':       self.n_undefined,
            'raw_responses':     self.raw_responses,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PromptInstance":
        return cls(
            dataset_idx       = d['dataset_idx'],
            wiki_bio_test_idx = d['wiki_bio_test_idx'],
            sentences         = d['sentences'],
            annotations       = d['annotations'],
            score_matrix      = d['score_matrix'],
            sentence_scores   = d['sentence_scores'],
            passage_score     = d['passage_score'],
            n_undefined       = d['n_undefined'],
            raw_responses     = d.get('raw_responses', []),
        )
