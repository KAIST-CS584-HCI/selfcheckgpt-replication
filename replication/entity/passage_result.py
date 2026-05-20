from dataclasses import dataclass


@dataclass
class PassageScores:
    prompt: list[float] | None = None
    bert:   list[float] | None = None
    nli:    list[float] | None = None

    def get(self, variant: str) -> list[float] | None:
        return getattr(self, variant)

    def to_dict(self) -> dict:
        return {
            variant: scores
            for variant, scores in {
                'prompt': self.prompt,
                'bert':   self.bert,
                'nli':    self.nli,
            }.items()
            if scores is not None
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PassageScores":
        return cls(
            prompt = d.get('prompt'),
            bert   = d.get('bert'),
            nli    = d.get('nli'),
        )


@dataclass
class PassageResponses:
    prompt: list[list[str | None]] | None = None

    def get(self, variant: str) -> object:
        return getattr(self, variant)

    def to_dict(self) -> dict:
        return {
            variant: responses
            for variant, responses in {
                'prompt': self.prompt,
            }.items()
            if responses is not None
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PassageResponses":
        return cls(
            prompt = d.get('prompt'),
        )


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
    scores:            PassageScores
    responses:         PassageResponses

    def to_dict(self) -> dict:
        return {
            'dataset_idx':       self.dataset_idx,
            'wiki_bio_test_idx': self.wiki_bio_test_idx,
            'wiki_bio_text':     self.wiki_bio_text,
            'main_passage':      self.main_passage,
            'main_sentences':    self.main_sentences,
            'annotation':        self.annotation,
            'sample_passages':   self.sample_passages,
            'scores':            self.scores.to_dict(),
            'responses':         self.responses.to_dict(),
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
            scores            = PassageScores.from_dict(d['scores']),
            responses         = PassageResponses.from_dict(d['responses']),
        )
