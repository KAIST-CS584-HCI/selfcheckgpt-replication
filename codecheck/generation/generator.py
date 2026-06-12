from __future__ import annotations
import re

from codecheck.generation.api_retry import chat_with_retries
from codecheck.generation.concurrency import map_staggered

_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)

PROMPT_TEMPLATE = (
    "Complete the following Python function. "
    "Return only the complete function implementation, no explanation.\n\n{prompt}"
)


def extract_code(text: str | None) -> str:
    if not text:
        return ""
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()


def build_prompt(problem) -> str:
    return PROMPT_TEMPLATE.format(prompt=problem.prompt)


class CodeGenerator:
    def __init__(self, client, model: str, think: bool = False, max_workers: int | None = None) -> None:
        self.client = client
        self.model = model
        self.think = think          # chain-of-thought reasoning; off = far faster
        self.max_workers = max_workers

    def _complete(self, prompt: str, temperature: float) -> str:
        # Reasoning off by default (enable with --think): reasoning models otherwise spend
        # thousands of hidden tokens per trivial function (~20x slower); the text is
        # discarded. Retries absorb transient API failures over a long run.
        resp = chat_with_retries(
            self.client,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            think=self.think,
        )
        if not resp.choices:
            return ""
        return extract_code(resp.choices[0].message.content)

    def generate(self, problem, n_samples: int, on_unit=None) -> tuple[str, list[str]]:
        prompt = build_prompt(problem)
        temps = [0.0] + [1.0] * n_samples  # main at T=0, n samples at T=1
        # Fire the N+1 calls concurrently but staggered, to avoid bursting the rate limit.
        # on_unit ticks once per completed call for live per-problem progress.
        outs = map_staggered(lambda t: self._complete(prompt, t), temps,
                             max_workers=self.max_workers, on_done=on_unit)
        return outs[0], outs[1:]
