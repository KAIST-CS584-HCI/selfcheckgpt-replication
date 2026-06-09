import codecheck.dataset as ds
from codecheck.models import CodeProblem

FAKE = {
    "Mbpp/2": {
        "task_id": "Mbpp/2", "prompt": "def f(x):\n    'doc'\n", "entry_point": "f",
        "canonical_solution": "def f(x):\n    return x + 1\n",
        "base_input": [[1], [2]], "plus_input": [[3]], "atol": 0.0,
    }
}


def test_maps_evalplus_to_code_problems(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "get_mbpp_plus", lambda: FAKE)
    cache = tmp_path / "mbpp_plus.json"
    problems = ds.load_mbpp_plus(limit=1, cache_path=cache)
    assert problems == [CodeProblem(
        task_id="Mbpp/2", prompt="def f(x):\n    'doc'\n", entry_point="f",
        canonical_solution="def f(x):\n    return x + 1\n",
        inputs=[[1], [2], [3]], atol=0.0,
    )]
    assert cache.exists()
