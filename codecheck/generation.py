from __future__ import annotations
import re

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
    def __init__(self, client, model: str) -> None:
        self.client = client
        self.model = model

    def _complete(self, prompt: str, temperature: float) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            # Disable chain-of-thought: reasoning models (e.g. qwen3) otherwise
            # spend thousands of hidden tokens on a trivial function — ~20x slower
            # — and we discard the reasoning text anyway. OpenRouter extension.
            extra_body={"reasoning": {"enabled": False}},
        )
        if not resp.choices:
            return ""
        return extract_code(resp.choices[0].message.content)

    def generate(self, problem, n_samples: int) -> tuple[str, list[str]]:
        prompt = build_prompt(problem)
        main = self._complete(prompt, 0.0)
        samples = [self._complete(prompt, 1.0) for _ in range(n_samples)]
        return main, samples
