from dataclasses import dataclass, field


@dataclass
class PassageResult:
    """Unified prediction output for one passage, supporting Prompt / BERT / NLI scores."""
    dataset_idx:       int
    wiki_bio_test_idx: int
    main_passage:      str
    sample_passages:   list[str]
    main_sentences:    list[str]
    annotation:        list[str]
    prompt_scores:     list[float] | None = None
    bert_scores:       list[float] | None = None
    nli_scores:        list[float] | None = None
    prompt_responses:  list[list[str | None]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'dataset_idx':       self.dataset_idx,
            'wiki_bio_test_idx': self.wiki_bio_test_idx,
            'main_passage':      self.main_passage,
            'sample_passages':   self.sample_passages,
            'main_sentences':    self.main_sentences,
            'annotation':        self.annotation,
            'prompt_scores':     self.prompt_scores,
            'bert_scores':       self.bert_scores,
            'nli_scores':        self.nli_scores,
            'prompt_responses':  self.prompt_responses,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PassageResult":
        return cls(
            dataset_idx       = d['dataset_idx'],
            wiki_bio_test_idx = d['wiki_bio_test_idx'],
            main_passage      = d['main_passage'],
            sample_passages   = d['sample_passages'],
            main_sentences    = d['main_sentences'],
            annotation        = d['annotation'],
            prompt_scores     = d.get('prompt_scores'),
            bert_scores       = d.get('bert_scores'),
            nli_scores        = d.get('nli_scores'),
            prompt_responses  = d.get('prompt_responses', []),
        )
