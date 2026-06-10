from codecheck.models import CodeProblem, CodeResult
from codecheck.execution import run_batch_in_subprocess
from codecheck.pipeline import score_problem, save_results, load_results

PROBLEM = CodeProblem(task_id="t", prompt="", entry_point="f",
                      canonical_solution="def f(x):\n    return x + 1\n",
                      inputs=[[1], [2], [3]], atol=0.0)


class StubGen:
    def __init__(self, main, samples):
        self._main, self._samples = main, samples

    def generate(self, problem, n_samples):
        return self._main, self._samples


def test_correct_main_with_consistent_samples():
    gen = StubGen("def f(x):\n    return x + 1\n",
                  ["def f(x):\n    return x + 1\n", "def f(x):\n    return 1 + x\n"])
    res = score_problem(PROBLEM, gen, run_batch_in_subprocess, n_samples=2, timeout=5.0)
    assert res.is_correct is True
    assert res.exec_score == 0.0


def test_incorrect_main_with_divergent_samples():
    gen = StubGen("def f(x):\n    return x\n",
                  ["def f(x):\n    return x + 1\n", "def f(x):\n    return x + 1\n"])
    res = score_problem(PROBLEM, gen, run_batch_in_subprocess, n_samples=2, timeout=5.0)
    assert res.is_correct is False
    assert res.exec_score == 1.0


def test_save_and_load_roundtrip(tmp_path):
    results = [CodeResult("t", {"exec": 0.5}, True, "m", ["s"], 3)]
    path = tmp_path / "out.json"
    save_results(results, path)
    assert load_results(path) == results


from codecheck.prompt_score import PromptJudge
from tests.test_codecheck_prompt_score import FakeJudgeClient


def test_score_problem_fills_exec_and_prompt():
    gen = StubGen("def f(x):\n    return x + 1\n",
                  ["def f(x):\n    return x + 1\n", "def f(x):\n    return x + 1\n"])
    judge = PromptJudge(FakeJudgeClient(["No.", "No."]), model="m")  # both inconsistent -> 1.0
    res = score_problem(PROBLEM, gen, run_batch_in_subprocess, n_samples=2, timeout=5.0,
                        methods={"exec", "prompt"}, judge=judge)
    assert "exec" in res.scores and "prompt" in res.scores
    assert res.scores["prompt"] == 1.0
    assert res.n_inputs == 3


def test_prompt_only_skips_sample_execution():
    # samples are not valid Python; if score_problem executed them for exec scoring
    # nothing breaks (the harness captures errors), but we assert exec is absent and
    # correctness labeling (main vs canonical) still works.
    gen = StubGen("def f(x):\n    return x + 1\n",
                  ["@@@ not python @@@", "also not python"])
    judge = PromptJudge(FakeJudgeClient(["Yes.", "Yes."]), model="m")  # consistent -> 0.0
    res = score_problem(PROBLEM, gen, run_batch_in_subprocess, n_samples=2, timeout=5.0,
                        methods={"prompt"}, judge=judge)
    assert "exec" not in res.scores
    assert res.scores["prompt"] == 0.0
    assert res.is_correct is True
    assert res.n_inputs == 3


from codecheck.ast_score import ASTScorer


def test_score_problem_fills_ast():
    gen = StubGen("def f(x):\n    return x + 1\n",
                  ["def f(x):\n    return x + 1\n", "def f(x):\n    return x + 1\n"])
    res = score_problem(PROBLEM, gen, run_batch_in_subprocess, n_samples=2, timeout=5.0,
                        methods={"ast"}, ast_scorer=ASTScorer())
    assert "ast" in res.scores
    assert "exec" not in res.scores          # ast-only run skips sample execution
    assert res.scores["ast"] == 0.0          # samples structurally identical to main
    assert res.is_correct is True            # labeling still runs (main vs canonical)
    assert res.n_inputs == 3


def test_score_problem_ast_requires_scorer():
    import pytest
    gen = StubGen("def f(x):\n    return x + 1\n", ["def f(x):\n    return x + 1\n"])
    with pytest.raises(ValueError):
        score_problem(PROBLEM, gen, run_batch_in_subprocess, n_samples=1, timeout=5.0,
                      methods={"ast"})
