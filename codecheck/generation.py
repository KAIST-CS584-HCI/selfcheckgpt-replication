from __future__ import annotations
import re
from concurrent.futures import ThreadPoolExecutor

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
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            # Reasoning models (e.g. qwen3) otherwise spend thousands of hidden
            # tokens per trivial function (~20x slower); the text is discarded.
            # Off by default; enable with --think. OpenRouter extension.
            extra_body={"reasoning": {"enabled": self.think}},
        )
        if not resp.choices:
            return ""
        return extract_code(resp.choices[0].message.content)

    def generate(self, problem, n_samples: int) -> tuple[str, list[str]]:
        prompt = build_prompt(problem)
        temps = [0.0] + [1.0] * n_samples  # main at T=0, n samples at T=1
        # Fire the N+1 calls concurrently; ex.map preserves input order.
        with ThreadPoolExecutor(max_workers=self.max_workers or len(temps)) as ex:
            outs = list(ex.map(lambda t: self._complete(prompt, t), temps))
        return outs[0], outs[1:]
