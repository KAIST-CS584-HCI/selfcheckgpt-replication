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
    results = [CodeResult("t", {"exec": 0.5}, True, "m", ["s"],
                          count={"total": 3, "pass": 3, "fail": 0, "error": 0})]
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
    assert res.count == {"total": 3, "pass": 3, "fail": 0, "error": 0}


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
    assert res.count == {"total": 3, "pass": 3, "fail": 0, "error": 0}


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
    assert res.count == {"total": 3, "pass": 3, "fail": 0, "error": 0}


def test_score_problem_ast_requires_scorer():
    import pytest
    gen = StubGen("def f(x):\n    return x + 1\n", ["def f(x):\n    return x + 1\n"])
    with pytest.raises(ValueError):
        score_problem(PROBLEM, gen, run_batch_in_subprocess, n_samples=1, timeout=5.0,
                      methods={"ast"})


from codecheck.pipeline import run_dataset


class RaisingGen:
    """Generator that raises on a target task_id, else returns trivial valid code."""

    def __init__(self, fail_task_id, exc=None):
        self.fail_task_id = fail_task_id
        self.exc = exc or RuntimeError("boom")

    def generate(self, problem, n_samples):
        if problem.task_id == self.fail_task_id:
            raise self.exc
        code = "def f(x):\n    return x + 1\n"
        return code, [code]


def _problems(*ids):
    return [CodeProblem(task_id=i, prompt="", entry_point="f",
                        canonical_solution="def f(x):\n    return x + 1\n",
                        inputs=[[1], [2]], atol=0.0) for i in ids]


def test_run_dataset_skips_failing_problem_and_continues():
    gen = RaisingGen(fail_task_id="b")
    results = run_dataset(_problems("a", "b", "c"), gen, run_batch_in_subprocess,
                          n_samples=1, timeout=5.0)
    assert [r.task_id for r in results] == ["a", "c"]   # b skipped, run completed


def test_run_dataset_reports_failed_task_ids(caplog):
    gen = RaisingGen(fail_task_id="b")
    with caplog.at_level("WARNING", logger="codecheck.pipeline"):
        run_dataset(_problems("a", "b", "c"), gen, run_batch_in_subprocess,
                    n_samples=1, timeout=5.0)
    assert any("b" in r.message and "failed" in r.message for r in caplog.records)


def test_run_dataset_propagates_keyboard_interrupt():
    import pytest
    gen = RaisingGen(fail_task_id="a", exc=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        run_dataset(_problems("a", "b"), gen, run_batch_in_subprocess,
                    n_samples=1, timeout=5.0)


def test_score_problem_sets_prompt_and_is_error_false_for_clean_main():
    prob = CodeProblem(task_id="t", prompt="THE PROMPT", entry_point="f",
                       canonical_solution="def f(x):\n    return x + 1\n",
                       inputs=[[1], [2], [3]], atol=0.0)
    gen = StubGen("def f(x):\n    return x + 1\n", ["def f(x):\n    return x + 1\n"])
    res = score_problem(prob, gen, run_batch_in_subprocess, n_samples=1, timeout=5.0)
    assert res.prompt == "THE PROMPT"
    assert res.is_error is False


def test_score_problem_is_error_true_when_main_raises():
    gen = StubGen("def f(x):\n    return undefined_name\n", ["def f(x):\n    return x + 1\n"])
    res = score_problem(PROBLEM, gen, run_batch_in_subprocess, n_samples=1, timeout=5.0)
    assert res.is_error is True
    assert res.is_correct is False


def test_score_problem_count_reflects_partial_match():
    # correct on x in {1,3}, wrong VALUE on x == 2 -> 2 pass, 1 fail, 0 error
    gen = StubGen("def f(x):\n    return x + 1 if x != 2 else 0\n", ["def f(x):\n    return x + 1\n"])
    res = score_problem(PROBLEM, gen, run_batch_in_subprocess, n_samples=1, timeout=5.0)
    assert res.count == {"total": 3, "pass": 2, "fail": 1, "error": 0}
    assert res.is_correct is False


from codecheck.pipeline import append_result
import json as _json


def test_append_result_then_load_roundtrips_in_order(tmp_path):
    path = tmp_path / "r.jsonl"
    a = CodeResult("a", {"exec": 0.1}, True, "m", ["s"])
    b = CodeResult("b", {"exec": 0.9}, False, "m", ["s"])
    append_result(a, path)
    append_result(b, path)
    assert load_results(path) == [a, b]


def test_save_results_jsonl_roundtrip(tmp_path):
    path = tmp_path / "r.jsonl"
    results = [CodeResult("a", {"exec": 0.1}, True, "m", ["s"]),
               CodeResult("b", {"exec": 0.9}, False, "m", ["s"])]
    save_results(results, path)
    assert path.read_text().count("\n") == 2     # one line per result
    assert load_results(path) == results


def test_load_results_reads_legacy_json_array(tmp_path):
    path = tmp_path / "legacy.json"
    legacy = [{"task_id": "a", "scores": {"exec": 0.3}, "is_correct": True,
               "main_code": "m", "sample_codes": ["s"]}]
    path.write_text(_json.dumps(legacy))
    loaded = load_results(path)
    assert [r.task_id for r in loaded] == ["a"]


def test_load_results_skips_torn_final_line(tmp_path):
    path = tmp_path / "r.jsonl"
    good = CodeResult("a", {"exec": 0.1}, True, "m", ["s"])
    append_result(good, path)
    with open(path, "a") as f:
        f.write('{"task_id": "b", "scor')   # crash mid-write: torn line, no newline
    assert [r.task_id for r in load_results(path)] == ["a"]


def test_run_dataset_calls_on_result_per_success(tmp_path):
    seen = []
    gen = RaisingGen(fail_task_id="b")   # b fails, a and c succeed
    run_dataset(_problems("a", "b", "c"), gen, run_batch_in_subprocess,
                n_samples=1, timeout=5.0, on_result=seen.append)
    assert [r.task_id for r in seen] == ["a", "c"]   # not called for the failed problem
