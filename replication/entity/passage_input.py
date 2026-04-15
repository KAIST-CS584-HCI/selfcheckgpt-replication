from dataclasses import dataclass


@dataclass
class PassageInput:
    """One passage from dataset.json."""
    wiki_bio_test_idx: int
    sentences:         list[str]   # gpt3_sentences — the text to be evaluated
    sampled_passages:  list[str]   # gpt3_text_samples — 20 stochastic samples
    annotation:        list[str]   # human labels per sentence, carried to output
    gpt3_text:         str         # original (unsplit) GPT-3 generated passage
    wiki_bio_text:     str         # ground-truth Wikipedia biography text

    @classmethod
    def from_dict(cls, item: dict) -> "PassageInput":
        return cls(
            wiki_bio_test_idx = item['wiki_bio_test_idx'],
            sentences         = item['gpt3_sentences'],
            sampled_passages  = item['gpt3_text_samples'],
            annotation        = item['annotation'],
            gpt3_text         = item['gpt3_text'],
            wiki_bio_text     = item['wiki_bio_text'],
        )
