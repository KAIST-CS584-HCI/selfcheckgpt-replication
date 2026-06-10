from codecheck.prompt_score import build_judge_prompt, parse_judgment


def test_build_judge_prompt_includes_both_codes():
    p = build_judge_prompt("def f(): return 1", "def f(): return 2")
    assert "def f(): return 1" in p
    assert "def f(): return 2" in p
    assert "Yes" in p and "No" in p


def test_parse_yes_means_consistent_zero():
    score, matched = parse_judgment("Yes, the behavior is identical.")
    assert score == 0.0 and matched is True


def test_parse_no_means_inconsistent_one():
    score, matched = parse_judgment("No - it differs on negative inputs.")
    assert score == 1.0 and matched is True


def test_parse_na_is_half():
    score, matched = parse_judgment("N/A because the construct is unrelated.")
    assert score == 0.5 and matched is True


def test_parse_unmatched_is_half_and_unmatched_flag():
    score, matched = parse_judgment("I cannot determine this.")
    assert score == 0.5 and matched is False


def test_parse_empty_is_unmatched_half():
    score, matched = parse_judgment("")
    assert score == 0.5 and matched is False
