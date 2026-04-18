from dataclasses import dataclass


@dataclass
class PassageOriginalInstance:
    """One passage from dataset.json (original GPT-3 dataset)."""
    wiki_bio_test_idx: int
    gpt3_text:         str
    gpt3_sentences:    list[str]
    annotation:        list[str]
    gpt3_text_samples: list[str]
    wiki_bio_text:     str

    @classmethod
    def from_dict(cls, item: dict) -> "PassageOriginalInstance":
        return cls(
            wiki_bio_test_idx = item['wiki_bio_test_idx'],
            gpt3_text         = item['gpt3_text'],
            gpt3_sentences    = item['gpt3_sentences'],
            annotation        = item['annotation'],
            gpt3_text_samples = item['gpt3_text_samples'],
            wiki_bio_text     = item['wiki_bio_text'],
        )
