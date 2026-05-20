from dataclasses import dataclass


@dataclass
class PassageInstance:
    """One passage from a normalized replication dataset."""
    wiki_bio_test_idx: int
    wiki_bio_text:     str
    main_passage:      str
    main_sentences:    list[str]
    annotation:        list[str]
    sample_passages:   list[str]

    @classmethod
    def from_dict(cls, item: dict) -> "PassageInstance":
        return cls(
            wiki_bio_test_idx = item['wiki_bio_test_idx'],
            wiki_bio_text     = item['wiki_bio_text'],
            main_passage      = item['main_passage'],
            main_sentences    = item['main_sentences'],
            annotation        = item['annotation'],
            sample_passages   = item['sample_passages'],
        )
