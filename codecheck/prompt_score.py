from __future__ import annotations
import re
from concurrent.futures import ThreadPoolExecutor

JUDGE_TEMPLATE = (
    "Implementation:\n{main_code}\n\n"
    "Does the following construct from another implementation have behavior "
    "consistent with the implementation above?\n"
    "Construct:\n{sample_code}\n\n"
    "Answer Yes / No / N/A with a one-sentence justification."
)

# Map a matched answer token to an inconsistency score (higher = more likely incorrect).
_ANSWER = re.compile(r"\b(yes|no|n/?a)\b", re.IGNORECASE)


def build_judge_prompt(main_code: str, sample_code: str) -> str:
    return JUDGE_TEMPLATE.format(main_code=main_code, sample_code=sample_code)


def parse_judgment(text: str | None) -> tuple[float, bool]:
    """(inconsistency_score, matched). Yes->0.0, No->1.0, N/A->0.5.
    Unparseable / empty -> (0.5, False) so callers can count parse failures."""
    if not text:
        return 0.5, False
    m = _ANSWER.search(text)
    if not m:
        return 0.5, False
    tok = m.group(1).lower().replace("/", "")
    if tok == "yes":
        return 0.0, True
    if tok == "no":
        return 1.0, True
    return 0.5, True   # "na"


class PromptJudge:
    """LLM-as-judge consistency scorer. score() returns mean inconsistency
    over the samples; parse_failures accumulates unparseable judgments."""

    def __init__(self, client, model: str, think: bool = False, max_workers: int | None = None) -> None:
        self.client = client
        self.model = model
        self.think = think
        self.max_workers = max_workers
        self.parse_failures = 0

    def _judge_one(self, main_code: str, sample_code: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": build_judge_prompt(main_code, sample_code)}],
            temperature=0.0,
            extra_body={"reasoning": {"enabled": self.think}},
        )
        if not resp.choices:
            return ""
        return resp.choices[0].message.content or ""

    def score(self, main_code: str, sample_codes: list[str]) -> float:
        if not sample_codes:
            return 0.0
        with ThreadPoolExecutor(max_workers=self.max_workers or len(sample_codes)) as ex:
            raws = list(ex.map(lambda s: self._judge_one(main_code, s), sample_codes))
        values = []
        for raw in raws:
            value, matched = parse_judgment(raw)
            if not matched:
                self.parse_failures += 1
            values.append(value)
        return sum(values) / len(values)
