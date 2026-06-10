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
                   main_code="def f(x): return x", sample_codes=["def f(x): return x", "def f(x): return 0"])
    assert CodeResult.from_dict(r.to_dict()) == r
