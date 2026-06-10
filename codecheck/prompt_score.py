from __future__ import annotations
import re

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
