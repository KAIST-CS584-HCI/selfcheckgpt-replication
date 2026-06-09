from types import SimpleNamespace
from codecheck.generation import CodeGenerator, extract_code
from codecheck.models import CodeProblem

PROBLEM = CodeProblem(task_id="t", prompt="def f(x):\n    'add one'\n", entry_point="f",
                      canonical_solution="def f(x):\n    return x+1\n", inputs=[[1]], atol=0.0)


def test_extract_code_from_fence():
    assert extract_code("blah\n```python\ndef f(x):\n    return x\n```\n") == "def f(x):\n    return x"
    assert extract_code("def f(x):\n    return x") == "def f(x):\n    return x"


class FakeClient:
    def __init__(self):
        self.calls = []
        content = "```python\ndef f(x):\n    return x + 1\n```"
        msg = SimpleNamespace(content=content)
        self._resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._resp


def test_generate_uses_temperatures_0_and_1():
    client = FakeClient()
    gen = CodeGenerator(client, model="m")
    main, samples = gen.generate(PROBLEM, n_samples=3)
    assert main == "def f(x):\n    return x + 1"
    assert len(samples) == 3
    temps = [c["temperature"] for c in client.calls]
    assert temps == [0.0, 1.0, 1.0, 1.0]
