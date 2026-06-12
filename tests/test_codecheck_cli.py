from types import SimpleNamespace
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_evaluate_subcommand_prints_auc(tmp_path):
    results = [
        {"task_id": "a", "exec_score": 0.9, "is_correct": False, "main_code": "m", "sample_codes": ["s"]},
        {"task_id": "b", "exec_score": 0.1, "is_correct": True, "main_code": "m", "sample_codes": ["s"]},
    ]
    path = tmp_path / "res.json"
    path.write_text(json.dumps(results))
    out = subprocess.run(
        [sys.executable, "run_codecheck.py", "evaluate", "--results", str(path)],
        cwd=REPO, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "AUC-PR" in out.stdout
    assert "n=2" in out.stdout


def test_run_parser_accepts_method():
    from run_codecheck import build_parser
    args = build_parser().parse_args(["run", "--method", "all", "--limit", "2"])
    assert args.method == "all"


def test_run_parser_method_defaults_to_exec():
    from run_codecheck import build_parser
    args = build_parser().parse_args(["run", "--limit", "2"])
    assert args.method == "exec"


def test_run_parser_rejects_unknown_method():
    import pytest
    from run_codecheck import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--method", "bogus"])


def test_method_ast_is_accepted_by_parser():
    from run_codecheck import build_parser
    args = build_parser().parse_args(["run", "--method", "ast"])
    assert args.method == "ast"


def test_method_all_is_accepted_by_parser():
    from run_codecheck import build_parser
    args = build_parser().parse_args(["run", "--method", "all"])
    assert args.method == "all"


def test_dataset_defaults_to_mbpp():
    from run_codecheck import build_parser
    assert build_parser().parse_args(["run"]).dataset == "mbpp"


def test_dataset_humaneval_is_accepted():
    from run_codecheck import build_parser
    assert build_parser().parse_args(["run", "--dataset", "humaneval"]).dataset == "humaneval"


def test_dataset_rejects_unknown():
    import pytest
    from run_codecheck import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--dataset", "bogus"])


def test_run_dispatches_to_humaneval_loader(monkeypatch, tmp_path):
    # --dataset humaneval must call load_human_eval_plus, not load_mbpp_plus.
    import codecheck.dataset as ds
    import run_codecheck
    from types import SimpleNamespace

    called = {}
    monkeypatch.setattr(ds, "load_human_eval_plus", lambda **kw: (called.setdefault("he", kw), [])[1])
    monkeypatch.setattr(ds, "load_mbpp_plus", lambda **kw: (called.setdefault("mbpp", kw), [])[1])
    # _setup_run_logging sets propagate=False on the codecheck logger; no-op it so this
    # test doesn't poison caplog for later logging tests.
    monkeypatch.setattr(run_codecheck, "_setup_run_logging", lambda **kw: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-dummy")
    # stop after loader dispatch: no problems -> "nothing to do" returns before any API call
    args = SimpleNamespace(dataset="humaneval", limit=2, index=None, randomize=False, seed=None,
                           n=2, timeout=5.0, think=False, verbose=False, method="exec",
                           ast_metric="jaccard", output=str(tmp_path / "o.json"))
    run_codecheck._cmd_run(args)
    assert "he" in called and "mbpp" not in called


def _judge_template_for_dataset(monkeypatch, tmp_path, dataset):
    """Build a prompt run for the given dataset and capture the template handed to PromptJudge."""
    import codecheck.dataset as ds
    import codecheck.score.prompt as ps
    import run_codecheck
    from types import SimpleNamespace

    monkeypatch.setattr(ds, "load_human_eval_plus", lambda **kw: [])
    monkeypatch.setattr(ds, "load_mbpp_plus", lambda **kw: [])
    monkeypatch.setattr(run_codecheck, "_setup_run_logging", lambda **kw: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-dummy")
    captured = {}

    def fake_judge(*a, **kw):
        captured["template"] = kw.get("template")
        return object()
    # _cmd_run does `from codecheck.score.prompt import PromptJudge` at call time, so
    # patching the attribute on that module is enough.
    monkeypatch.setattr(ps, "PromptJudge", fake_judge)
    args = SimpleNamespace(dataset=dataset, limit=2, index=None, randomize=False, seed=None,
                           n=2, timeout=5.0, think=False, verbose=False, method="prompt",
                           ast_metric="jaccard", output=str(tmp_path / "o.json"))
    run_codecheck._cmd_run(args)
    return captured["template"]


def test_humaneval_run_selects_divergence_judge_template(monkeypatch, tmp_path):
    from codecheck.score.prompt import HUMANEVAL_JUDGE_TEMPLATE
    assert _judge_template_for_dataset(monkeypatch, tmp_path, "humaneval") == HUMANEVAL_JUDGE_TEMPLATE


def test_mbpp_run_selects_default_judge_template(monkeypatch, tmp_path):
    from codecheck.score.prompt import JUDGE_TEMPLATE
    assert _judge_template_for_dataset(monkeypatch, tmp_path, "mbpp") == JUDGE_TEMPLATE


def test_ast_metric_defaults_to_jaccard():
    from run_codecheck import build_parser
    args = build_parser().parse_args(["run"])
    assert args.ast_metric == "jaccard"


def test_ast_metric_accepts_jaccard():
    from run_codecheck import build_parser
    args = build_parser().parse_args(["run", "--ast-metric", "jaccard"])
    assert args.ast_metric == "jaccard"


def test_ast_metric_rejects_unknown():
    import pytest
    from run_codecheck import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--ast-metric", "bogus"])


def test_format_evaluation_lists_each_method():
    from run_codecheck import format_evaluation
    from codecheck.models import CodeResult
    results = [
        CodeResult("a", {"exec": 0.9, "prompt": 0.2}, False, "m", []),
        CodeResult("b", {"exec": 0.1, "prompt": 0.8}, True, "m", []),
    ]
    text = format_evaluation(results)
    assert "exec" in text and "prompt" in text
    assert "AUC-PR" in text
    assert "baseline" in text.lower()


def test_format_evaluation_handles_mixed_method_results():
    from run_codecheck import format_evaluation
    from codecheck.models import CodeResult
    # one record has both methods, one is exec-only -> must not raise
    results = [
        CodeResult("a", {"exec": 0.9, "prompt": 0.8}, False, "m", []),
        CodeResult("b", {"exec": 0.1}, True, "m", []),
    ]
    text = format_evaluation(results)        # would KeyError before the fix
    assert "exec" in text and "prompt" in text


def test_index_arg_parses():
    from run_codecheck import build_parser
    assert build_parser().parse_args(["run", "--index", "0"]).index == 0


def test_limit_defaults_to_none():
    from run_codecheck import build_parser
    assert build_parser().parse_args(["run"]).limit is None


def test_resolve_selection_bare_run_is_entire_dataset():
    # No --limit -> limit=None, which the loaders treat as the whole dataset.
    from run_codecheck import build_parser, _resolve_selection
    args = build_parser().parse_args(["run"])
    assert _resolve_selection(args) == (None, None)


def test_resolve_selection_index_mode():
    from run_codecheck import build_parser, _resolve_selection
    args = build_parser().parse_args(["run", "--index", "3"])
    assert _resolve_selection(args) == (None, 3)


def test_resolve_selection_index_with_random_errors():
    import pytest
    from run_codecheck import build_parser, _resolve_selection
    args = build_parser().parse_args(["run", "--index", "0", "--random"])
    with pytest.raises(ValueError):
        _resolve_selection(args)


def test_resolve_selection_index_with_limit_errors():
    import pytest
    from run_codecheck import build_parser, _resolve_selection
    args = build_parser().parse_args(["run", "--index", "0", "--limit", "5"])
    with pytest.raises(ValueError):
        _resolve_selection(args)


def test_resolve_selection_index_with_seed_errors():
    import pytest
    from run_codecheck import build_parser, _resolve_selection
    args = build_parser().parse_args(["run", "--index", "0", "--seed", "1"])
    with pytest.raises(ValueError):
        _resolve_selection(args)


def test_method_code_bert_is_accepted_by_parser():
    from run_codecheck import build_parser
    assert build_parser().parse_args(["run", "--method", "code_bert"]).method == "code_bert"


def test_codebert_subcommand_parses_results():
    from run_codecheck import build_parser
    args = build_parser().parse_args(["codebert", "--results", "results/x.json"])
    assert args.results == "results/x.json"


def test_cmd_codebert_augments_results_in_place(tmp_path, monkeypatch):
    import codecheck.score.codebert as cbs
    from codecheck.models import CodeResult
    from codecheck.pipeline import save_results, load_results
    import run_codecheck

    class FakeScorer:
        def score(self, main_code, sample_codes):
            return 0.42

    monkeypatch.setattr(cbs, "CodeBERTScorer", FakeScorer)
    monkeypatch.setattr(cbs, "torch_available", lambda: True)  # fake scorer stands in for the model
    path = tmp_path / "r.json"
    save_results([CodeResult("a", {"exec": 0.1}, True, "def m(): pass", ["def s(): pass"])], path)

    run_codecheck._cmd_codebert(SimpleNamespace(results=str(path)))

    out = load_results(path)
    assert out[0].scores["code_bert"] == 0.42
    assert out[0].scores["exec"] == 0.1   # existing scores preserved


def test_cmd_codebert_exits_clean_when_torch_missing(tmp_path, monkeypatch):
    import codecheck.score.codebert as cbs
    import run_codecheck
    import pytest

    monkeypatch.setattr(cbs, "torch_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        run_codecheck._cmd_codebert(SimpleNamespace(results=str(tmp_path / "missing.json")))
    assert "torch" in str(exc.value)   # fails fast with a clear message, not a traceback
