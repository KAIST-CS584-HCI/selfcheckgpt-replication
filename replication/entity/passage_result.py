from dataclasses import dataclass


@dataclass
class PassageResult:
    """Unified prediction output for one passage, supporting Prompt / BERT / NLI scores."""
    dataset_idx:       int
    wiki_bio_test_idx: int
    wiki_bio_text:     str
    main_passage:      str
    main_sentences:    list[str]
    annotation:        list[str]
    sample_passages:   list[str]
    result:            dict[str, dict]

    def to_dict(self) -> dict:
        return {
            'dataset_idx':       self.dataset_idx,
            'wiki_bio_test_idx': self.wiki_bio_test_idx,
            'wiki_bio_text':     self.wiki_bio_text,
            'main_passage':      self.main_passage,
            'main_sentences':    self.main_sentences,
            'annotation':        self.annotation,
            'sample_passages':   self.sample_passages,
            'result':            self.result,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PassageResult":
        return cls(
            dataset_idx       = d['dataset_idx'],
            wiki_bio_test_idx = d['wiki_bio_test_idx'],
            wiki_bio_text     = d['wiki_bio_text'],
            main_passage      = d['main_passage'],
            main_sentences    = d['main_sentences'],
            annotation        = d['annotation'],
            sample_passages   = d['sample_passages'],
            result            = d['result'],
        )
