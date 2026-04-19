from dataclasses import dataclass


@dataclass
class GeneratedTextInstance:
    """One entry from a generated-text dataset (e.g. dataset-generated-*.json)."""
    example_id:       int
    concept:          str
    prompt:           str
    main_response:    str
    sampled_passages: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> "GeneratedTextInstance":
        return cls(
            example_id       = d['example_id'],
            concept          = d['concept'],
            prompt           = d['prompt'],
            main_response    = d['main_response'],
            sampled_passages = d['sampled_passages'],
        )

    def to_dict(self) -> dict:
        return {
            'example_id':       self.example_id,
            'concept':          self.concept,
            'prompt':           self.prompt,
            'main_response':    self.main_response,
            'sampled_passages': self.sampled_passages,
        }
