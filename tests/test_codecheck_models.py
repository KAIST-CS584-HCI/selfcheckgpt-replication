from codecheck.models import CodeProblem, CodeResult


def test_code_problem_roundtrip():
    p = CodeProblem(
        task_id="Mbpp/2", prompt="def f(x):\n    'doc'\n", entry_point="f",
        canonical_solution="def f(x):\n    return x + 1\n",
        inputs=[[1], [2]], atol=1e-6,
    )
    assert CodeProblem.from_dict(p.to_dict()) == p


def test_code_result_roundtrip():
    r = CodeResult(task_id="Mbpp/2", scores={"exec": 0.4}, is_correct=True,
                   main_code="def f(x): return x", sample_codes=["def f(x): return x", "def f(x): return 0"],
                   n_inputs=106)
    assert CodeResult.from_dict(r.to_dict()) == r


def test_code_result_prompt_responses_roundtrip():
    r = CodeResult(task_id="Mbpp/2", scores={"prompt": 0.5}, is_correct=True,
                   main_code="m", sample_codes=["s1", "s2"], n_inputs=3,
                   prompt_responses=["Yes.", "No."])
    assert CodeResult.from_dict(r.to_dict()) == r


def test_code_result_omits_prompt_responses_when_absent():
    # exec-only runs carry no judge text; the key stays out of the JSON
    r = CodeResult("t", {"exec": 0.4}, True, "m", ["s"])
    assert "prompt_responses" not in r.to_dict()
    assert CodeResult.from_dict(r.to_dict()).prompt_responses is None


def test_code_result_n_inputs_defaults_for_legacy_json():
    # iteration-1 artifacts had no n_inputs key
    legacy = {"task_id": "t", "exec_score": 0.3, "is_correct": False,
              "main_code": "m", "sample_codes": ["s"]}
    assert CodeResult.from_dict(legacy).n_inputs == 0


def test_coderesult_scores_dict_roundtrip():
    r = CodeResult("t", {"exec": 0.3, "prompt": 0.7}, False, "main", ["s1"])
    d = r.to_dict()
    assert d["scores"] == {"exec": 0.3, "prompt": 0.7}
    back = CodeResult.from_dict(d)
    assert back.scores == {"exec": 0.3, "prompt": 0.7}
    assert back.is_correct is False


def test_coderesult_exec_score_property():
    r = CodeResult("t", {"exec": 0.42}, True, "m", [])
    assert r.exec_score == 0.42


def test_coderesult_from_legacy_exec_score_key():
    # iteration-1 artifacts stored a bare "exec_score" key, no "scores"
    legacy = {"task_id": "t", "exec_score": 0.358, "is_correct": False,
              "main_code": "m", "sample_codes": ["s"]}
    r = CodeResult.from_dict(legacy)
    assert r.scores == {"exec": 0.358}
    assert r.exec_score == 0.358
