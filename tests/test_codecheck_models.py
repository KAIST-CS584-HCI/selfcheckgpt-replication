from codecheck.models import CodeProblem, CodeResult


def test_code_problem_roundtrip():
    p = CodeProblem(
        task_id="Mbpp/2", prompt="def f(x):\n    'doc'\n", entry_point="f",
        canonical_solution="def f(x):\n    return x + 1\n",
        inputs=[[1], [2]], atol=1e-6,
    )
    assert CodeProblem.from_dict(p.to_dict()) == p


def test_code_result_roundtrip():
    r = CodeResult(task_id="Mbpp/2", exec_score=0.4, is_correct=True,
                   main_code="def f(x): return x", sample_codes=["def f(x): return x", "def f(x): return 0"],
                   n_inputs=106)
    assert CodeResult.from_dict(r.to_dict()) == r


def test_code_result_n_inputs_defaults_for_legacy_json():
    # iteration-1 artifacts had no n_inputs key
    legacy = {"task_id": "t", "exec_score": 0.3, "is_correct": False,
              "main_code": "m", "sample_codes": ["s"]}
    assert CodeResult.from_dict(legacy).n_inputs == 0
