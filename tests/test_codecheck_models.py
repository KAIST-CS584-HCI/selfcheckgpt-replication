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
                   count={"total": 106, "pass": 100, "fail": 6, "error": 0})
    assert CodeResult.from_dict(r.to_dict()) == r


def test_code_result_prompt_responses_roundtrip():
    r = CodeResult(task_id="Mbpp/2", scores={"prompt": 0.5}, is_correct=True,
                   main_code="m", sample_codes=["s1", "s2"],
                   count={"total": 3, "pass": 2, "fail": 1, "error": 0},
                   prompt_responses=["Yes.", "No."])
    assert CodeResult.from_dict(r.to_dict()) == r


def test_code_result_omits_prompt_responses_when_absent():
    # exec-only runs carry no judge text; the key stays out of the JSON
    r = CodeResult("t", {"exec": 0.4}, True, "m", ["s"])
    assert "prompt_responses" not in r.to_dict()
    assert CodeResult.from_dict(r.to_dict()).prompt_responses is None


def test_code_result_count_defaults_for_legacy_json():
    # older artifacts had no count key
    legacy = {"task_id": "t", "exec_score": 0.3, "is_correct": False,
              "main_code": "m", "sample_codes": ["s"]}
    assert CodeResult.from_dict(legacy).count == {"total": 0, "pass": 0, "fail": 0, "error": 0}


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


def test_code_result_prompt_and_is_error_roundtrip():
    r = CodeResult(task_id="Mbpp/2", scores={"exec": 0.4}, is_correct=False,
                   main_code="m", sample_codes=["s"],
                   count={"total": 3, "pass": 2, "fail": 0, "error": 1},
                   prompt="def f(x):\n    'doc'\n", is_error=True)
    back = CodeResult.from_dict(r.to_dict())
    assert back.prompt == "def f(x):\n    'doc'\n"
    assert back.is_error is True
    assert back == r


def test_code_result_to_dict_includes_prompt_and_is_error():
    r = CodeResult("t", {"exec": 0.4}, True, "m", ["s"], prompt="P", is_error=False)
    d = r.to_dict()
    assert d["prompt"] == "P"
    assert d["is_error"] is False


def test_code_result_to_dict_has_count_not_n_inputs_or_passed():
    r = CodeResult("t", {"exec": 0.4}, True, "m", ["s"],
                   count={"total": 3, "pass": 3, "fail": 0, "error": 0})
    d = r.to_dict()
    assert d["count"] == {"total": 3, "pass": 3, "fail": 0, "error": 0}
    assert "n_inputs" not in d
    assert "passed" not in d


def test_code_result_defaults_for_legacy_json_without_prompt_or_is_error():
    legacy = {"task_id": "t", "scores": {"exec": 0.3}, "is_correct": False,
              "main_code": "m", "sample_codes": ["s"]}
    r = CodeResult.from_dict(legacy)
    assert r.prompt == ""
    assert r.is_error is False
