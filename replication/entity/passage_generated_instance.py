from dataclasses import dataclass


@dataclass
class PassageGeneratedInstance:
    """One passage from dataset-generated-samples-gpt-3.5-turbo.json."""
    wiki_bio_test_idx: int
    main_passage:      str
    main_sentences:    list[str]
    annotation:        list[str]
    sample_passages:   list[str]
    wiki_bio_text:     str

    @classmethod
    def from_dict(cls, item: dict) -> "PassageGeneratedInstance":
        return cls(
            wiki_bio_test_idx = item['wiki_bio_test_idx'],
            main_passage      = item['main_passage'],
            main_sentences    = item['main_sentences'],
            annotation        = item['annotation'],
            sample_passages   = item['sample_passages'],
            wiki_bio_text     = item['wiki_bio_text'],
        )
