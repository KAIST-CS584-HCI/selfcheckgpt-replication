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


def test_index_selects_single_problem_in_order(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "get_mbpp_plus", lambda: _fake_many(10))
    problems = ds.load_mbpp_plus(index=0, cache_path=tmp_path / "c.json")
    assert [p.task_id for p in problems] == ["Mbpp/0"]


def test_index_selects_nth_problem(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "get_mbpp_plus", lambda: _fake_many(10))
    problems = ds.load_mbpp_plus(index=4, cache_path=tmp_path / "c.json")
    assert [p.task_id for p in problems] == ["Mbpp/4"]


def test_index_out_of_range_raises(monkeypatch, tmp_path):
    import pytest
    monkeypatch.setattr(ds, "get_mbpp_plus", lambda: _fake_many(10))
    with pytest.raises(IndexError):
        ds.load_mbpp_plus(index=10, cache_path=tmp_path / "c.json")


def test_index_negative_raises(monkeypatch, tmp_path):
    import pytest
    monkeypatch.setattr(ds, "get_mbpp_plus", lambda: _fake_many(10))
    with pytest.raises(IndexError):
        ds.load_mbpp_plus(index=-1, cache_path=tmp_path / "c.json")


def test_cache_path_accepts_str(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "get_mbpp_plus", lambda: FAKE)
    cache = str(tmp_path / "c.json")           # str, not Path
    problems = ds.load_mbpp_plus(limit=1, cache_path=cache)
    assert problems and __import__("os").path.exists(cache)


# --- HumanEval+ ---

# HumanEval+ splits the reference: `prompt` is the signature+docstring, `canonical_solution`
# is the body only. The loader must assemble prompt + body into a runnable function.
HE_FAKE = {
    "HumanEval/0": {
        "task_id": "HumanEval/0",
        "prompt": "def f(x):\n    'doc'\n",
        "entry_point": "f",
        "canonical_solution": "    return x + 1\n",   # body only, no def line
        "base_input": [[1], [2]], "plus_input": [[3]], "atol": 0.0,
    }
}


def _he_fake_many(n):
    return {
        f"HumanEval/{i}": {
            "task_id": f"HumanEval/{i}", "prompt": "def f():\n    'd'\n", "entry_point": "f",
            "canonical_solution": "    return 0\n", "base_input": [[i]], "plus_input": [], "atol": 0.0,
        }
        for i in range(n)
    }


def test_human_eval_assembles_canonical_from_prompt_plus_body(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "get_human_eval_plus", lambda: HE_FAKE)
    problems = ds.load_human_eval_plus(limit=1, cache_path=tmp_path / "he.json")
    assert problems == [CodeProblem(
        task_id="HumanEval/0", prompt="def f(x):\n    'doc'\n", entry_point="f",
        canonical_solution="def f(x):\n    'doc'\n    return x + 1\n",  # prompt + body
        inputs=[[1], [2], [3]], atol=0.0,
    )]


def test_human_eval_reuses_selection_and_index(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "get_human_eval_plus", lambda: _he_fake_many(10))
    first3 = ds.load_human_eval_plus(limit=3, randomize=False, cache_path=tmp_path / "a.json")
    assert [p.task_id for p in first3] == ["HumanEval/0", "HumanEval/1", "HumanEval/2"]
    one = ds.load_human_eval_plus(index=4, cache_path=tmp_path / "b.json")
    assert [p.task_id for p in one] == ["HumanEval/4"]


def test_human_eval_index_out_of_range_raises(monkeypatch, tmp_path):
    import pytest
    monkeypatch.setattr(ds, "get_human_eval_plus", lambda: _he_fake_many(10))
    with pytest.raises(IndexError):
        ds.load_human_eval_plus(index=10, cache_path=tmp_path / "c.json")


# --- CodeHaluEval (stdin/stdout) ---

import json as _json

# One row per (task_id, test_case). `input`/`output` are RAW stdin/stdout strings; `solutions`
# is a JSON-encoded list string (`''` when none); `task_id` is an int. Rows with a non-empty
# `fn_name` are the call-style tasks the stdio harness must skip.
CHE_FAKE = [
    {"task_id": 10, "question": "read n, print n+1", "fn_name": None,
     "solutions": _json.dumps(["print(int(input())+1)"]),
     "input": "1\n", "output": "2\n"},
    {"task_id": 10, "question": "read n, print n+1", "fn_name": None,
     "solutions": _json.dumps(["print(int(input())+1)"]),
     "input": "5\n", "output": "6 \r\n"},                          # cosmetic ws -> normalized
    {"task_id": 11, "question": "fn task", "fn_name": "solve",
     "solutions": _json.dumps(["def solve(): pass"]),
     "input": "x", "output": "y"},
]


def test_codehalu_groups_filters_and_builds_expected(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "_load_codehalu_rows", lambda: CHE_FAKE)
    problems = ds.load_codehalu_eval(cache_path=tmp_path / "che.json")
    assert len(problems) == 1                       # task 11 (fn_name) filtered out
    p = problems[0]
    assert p.task_id == "CodeHalu/10"               # int task_id -> prefixed string
    assert p.entry_point == ""                      # stdio, no callable
    assert p.prompt == "read n, print n+1"
    assert p.canonical_solution == "print(int(input())+1)"
    assert p.inputs == ["1\n", "5\n"]               # both test cases, raw stdin
    assert p.expected == [("value", "2"), ("value", "6")]   # normalized stdout, ground truth
    assert p.atol == 0.0


def test_codehalu_empty_solutions_is_tolerated(monkeypatch, tmp_path):
    rows = [{"task_id": 1, "question": "q", "fn_name": None,
             "solutions": "", "input": "a\n", "output": "a\n"}]
    monkeypatch.setattr(ds, "_load_codehalu_rows", lambda: rows)
    problems = ds.load_codehalu_eval(cache_path=tmp_path / "c.json")
    assert problems[0].canonical_solution == ""     # empty solutions -> no canonical, no crash


def test_codehalu_list_typed_input_output_joined_with_newlines(monkeypatch, tmp_path):
    # A few rows store stdin/stdout as a list of lines instead of a raw string.
    rows = [{"task_id": 5, "question": "q", "fn_name": None,
             "solutions": _json.dumps(["pass"]),
             "input": ["1", "64 99"], "output": ["1337"]}]
    monkeypatch.setattr(ds, "_load_codehalu_rows", lambda: rows)
    p = ds.load_codehalu_eval(cache_path=tmp_path / "c.json")[0]
    assert p.inputs == ["1\n64 99"]
    assert p.expected == [("value", "1337")]


def test_codehalu_caps_cases_per_task(monkeypatch, tmp_path):
    rows = [
        {"task_id": 7, "question": "q", "fn_name": None,
         "solutions": _json.dumps(["pass"]), "input": f"{i}\n", "output": f"{i}\n"}
        for i in range(10)
    ]
    monkeypatch.setattr(ds, "_load_codehalu_rows", lambda: rows)
    problems = ds.load_codehalu_eval(max_cases=3, cache_path=tmp_path / "c.json")
    assert len(problems[0].inputs) == 3
    assert len(problems[0].expected) == 3


def test_codehalu_reuses_limit_selection(monkeypatch, tmp_path):
    rows = [
        {"task_id": i, "question": "q", "fn_name": None,
         "solutions": _json.dumps(["pass"]), "input": "a", "output": "a"}
        for i in range(5)
    ]
    monkeypatch.setattr(ds, "_load_codehalu_rows", lambda: rows)
    problems = ds.load_codehalu_eval(limit=2, randomize=False, cache_path=tmp_path / "c.json")
    assert [p.task_id for p in problems] == ["CodeHalu/0", "CodeHalu/1"]
