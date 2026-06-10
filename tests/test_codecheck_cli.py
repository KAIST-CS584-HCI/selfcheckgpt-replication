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
        build_parser().parse_args(["run", "--method", "ast"])
