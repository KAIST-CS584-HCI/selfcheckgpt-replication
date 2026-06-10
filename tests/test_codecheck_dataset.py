import codecheck.dataset as ds
from codecheck.models import CodeProblem

FAKE = {
    "Mbpp/2": {
        "task_id": "Mbpp/2", "prompt": "def f(x):\n    'doc'\n", "entry_point": "f",
        "canonical_solution": "def f(x):\n    return x + 1\n",
        "base_input": [[1], [2]], "plus_input": [[3]], "atol": 0.0,
    }
}


def _fake_many(n):
    return {
        f"Mbpp/{i}": {
            "task_id": f"Mbpp/{i}", "prompt": "p", "entry_point": "f",
            "canonical_solution": "def f(): pass", "base_input": [[i]], "plus_input": [], "atol": 0.0,
        }
        for i in range(n)
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


def test_no_random_takes_first_n_in_order(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "get_mbpp_plus", lambda: _fake_many(10))
    problems = ds.load_mbpp_plus(limit=3, randomize=False, cache_path=tmp_path / "c.json")
    assert [p.task_id for p in problems] == ["Mbpp/0", "Mbpp/1", "Mbpp/2"]


def test_random_default_is_a_subset_of_size_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "get_mbpp_plus", lambda: _fake_many(10))
    problems = ds.load_mbpp_plus(limit=3, cache_path=tmp_path / "c.json")
    ids = [p.task_id for p in problems]
    assert len(ids) == 3
    assert len(set(ids)) == 3  # no duplicates
    assert all(i in {f"Mbpp/{j}" for j in range(10)} for i in ids)


def test_same_seed_is_reproducible(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "get_mbpp_plus", lambda: _fake_many(50))
    a = ds.load_mbpp_plus(limit=5, seed=7, cache_path=tmp_path / "a.json")
    b = ds.load_mbpp_plus(limit=5, seed=7, cache_path=tmp_path / "b.json")
    assert [p.task_id for p in a] == [p.task_id for p in b]
