from __future__ import annotations
import re

from codecheck.generation.api_retry import APIRetriesExhausted, chat_with_retries
from codecheck.generation.concurrency import map_staggered

JUDGE_TEMPLATE = (
    "Implementation:\n{main_code}\n\n"
    "Does the following construct from another implementation have behavior "
    "consistent with the implementation above?\n"
    "Construct:\n{sample_code}\n\n"
    "Answer Yes / No / N/A with a one-sentence justification."
)

# A sharper, still oracle-free variant used for HumanEval+: a capable model writes
# near-identical samples on canonical problems, so the default "consistent?" prompt lazily
# answers "Yes, identical" and scores 0.0 even for wrong mains. This framing makes the judge
# hunt for an edge-case input on which the two implementations diverge — same Yes/No/N-A
# mapping (Yes = identical behavior = consistent = 0.0). Validated to roughly triple the
# incorrect-vs-correct separation on a real HumanEval+ run; no spec/expected-output is shown.
HUMANEVAL_JUDGE_TEMPLATE = (
    "Two Python implementations of the same function:\n\n"
    "Implementation A:\n{main_code}\n\n"
    "Implementation B:\n{sample_code}\n\n"
    "Ignoring comments and docstrings, would A and B return the SAME result on EVERY input, "
    "including edge cases (empty, zero, negatives, large, duplicates, unusual types)? Look "
    "hard for any single input on which they would differ or one would error/loop.\n"
    "Answer Yes (always identical behavior) / No (they can differ) / N/A, with a one-sentence reason."
)

# CodeHaluEval is whole-program stdin->stdout (Codeforces-style), not a function call. Same
# oracle-free, divergence-seeking framing as the HumanEval template, but over programs that
# read stdin and print to stdout. Same Yes/No/N-A mapping (Yes = identical output = 0.0).
CODEHALU_JUDGE_TEMPLATE = (
    "Two Python programs that read from stdin and write to stdout for the same task:\n\n"
    "Program A:\n{main_code}\n\n"
    "Program B:\n{sample_code}\n\n"
    "Ignoring comments, would A and B print the SAME output for EVERY valid stdin, including "
    "edge cases (empty, zero, negatives, large, duplicates, boundary sizes)? Look hard for any "
    "single stdin on which their output would differ or one would error/loop.\n"
    "Answer Yes (always identical output) / No (they can differ) / N/A, with a one-sentence reason."
)

# Map a matched answer token to an inconsistency score (higher = more likely incorrect).
_ANSWER = re.compile(r"\b(yes|no|n/?a)\b", re.IGNORECASE)


def build_judge_prompt(main_code: str, sample_code: str, template: str = JUDGE_TEMPLATE) -> str:
    return template.format(main_code=main_code, sample_code=sample_code)


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

    def __init__(self, client, model: str, think: bool = False, max_workers: int | None = None,
                 template: str = JUDGE_TEMPLATE) -> None:
        self.client = client
        self.model = model
        self.think = think
        self.max_workers = max_workers
        self.template = template
        self.parse_failures = 0

    def _judge_one(self, main_code: str, sample_code: str) -> str:
        try:
            resp = chat_with_retries(
                self.client,
                model=self.model,
                messages=[{"role": "user", "content": build_judge_prompt(main_code, sample_code, self.template)}],
                temperature=0.0,
                think=self.think,
            )
        except APIRetriesExhausted:
            # A persistently-failing judge call is an unusable judgment, not a fatal error:
            # return "" so it counts as a parse failure and the run survives the bad sample.
            return ""
        if not resp.choices:
            return ""
        return resp.choices[0].message.content or ""

    def evaluate(self, main_code: str, sample_codes: list[str], on_unit=None) -> tuple[float, list[str]]:
        """(mean_inconsistency, raw_responses). raw_responses runs parallel to
        sample_codes so callers can record per-sample judge text for variance analysis.
        on_unit ticks once per completed judge call for live per-problem progress."""
        if not sample_codes:
            return 0.0, []
        raws = map_staggered(lambda s: self._judge_one(main_code, s), sample_codes,
                             max_workers=self.max_workers, on_done=on_unit)
        values = []
        for raw in raws:
            value, matched = parse_judgment(raw)
            if not matched:
                self.parse_failures += 1
            values.append(value)
        return sum(values) / len(values), raws

    def score(self, main_code: str, sample_codes: list[str]) -> float:
        return self.evaluate(main_code, sample_codes)[0]
