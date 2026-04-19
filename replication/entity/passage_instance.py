from dataclasses import dataclass


@dataclass
class PassageInstance:
    """One passage from dataset-generated-samples-gpt-3.5-turbo.json."""
    wiki_bio_test_idx: int
    main_passage:      str
    main_sentences:    list[str]
    annotation:        list[str]
    sample_passages:   list[str]
    wiki_bio_text:     str

    @classmethod
    def from_dict(cls, item: dict) -> "PassageInstance":
        return cls(
            wiki_bio_test_idx = item['wiki_bio_test_idx'],
            main_passage      = item['main_passage'],
            main_sentences    = item['main_sentences'],
            annotation        = item['annotation'],
            sample_passages   = item['sample_passages'],
            wiki_bio_text     = item['wiki_bio_text'],
        )

    @classmethod
    def from_original_dict(cls, item: dict) -> "PassageInstance":
        return cls.from_dict({
            **item,
            'main_passage':    item['gpt3_text'],
            'main_sentences':  item['gpt3_sentences'],
            'sample_passages': item['gpt3_text_samples'],
        })
