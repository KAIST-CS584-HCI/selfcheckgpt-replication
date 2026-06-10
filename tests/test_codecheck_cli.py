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
    args = build_parser().parse_args(["run", "--method", "both", "--limit", "2"])
    assert args.method == "both"


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
