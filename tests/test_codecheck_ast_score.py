from codecheck.ast_score import ast_fingerprint, ast_dissimilarity


def test_fingerprint_is_invariant_to_variable_renaming():
    a = ast_fingerprint("def f(x):\n    return x + 1\n")
    b = ast_fingerprint("def f(y):\n    return y + 1\n")
    assert a == b


def test_fingerprint_is_invariant_to_literal_values():
    a = ast_fingerprint("def f(x):\n    return x + 1\n")
    b = ast_fingerprint("def f(x):\n    return x + 99\n")
    assert a == b


def test_fingerprint_returns_none_on_syntax_error():
    assert ast_fingerprint("@@@ not python @@@") is None


def test_fingerprint_returns_none_on_empty_or_bodyless_code():
    # A blank generation is a non-answer, not a structure to compare. Empty,
    # whitespace-only, and comment-only all parse to a body-less module.
    assert ast_fingerprint("") is None
    assert ast_fingerprint("   \n  \t\n") is None
    assert ast_fingerprint("# just a comment\n") is None


def test_fingerprint_returns_none_on_non_str_input():
    # The SyntaxError-only guard let None through as an uncaught TypeError.
    assert ast_fingerprint(None) is None


def test_identical_structure_has_zero_dissimilarity():
    a = ast_fingerprint("def f(x):\n    return x + 1\n")
    assert ast_dissimilarity(a, a) == 0.0


def test_different_structure_has_positive_dissimilarity():
    a = ast_fingerprint("def f(x):\n    return x + 1\n")
    b = ast_fingerprint("def f(x):\n    for i in range(x):\n        print(i)\n")
    d = ast_dissimilarity(a, b)
    assert 0.0 < d <= 1.0


from codecheck.ast_score import ASTScorer


def test_scorer_mean_dissimilarity_over_samples():
    main = "def f(x):\n    return x + 1\n"
    scorer = ASTScorer()
    # identical sample -> 0.0; structurally different sample -> > 0.0; mean is between.
    score, per_sample = scorer.evaluate(main, [main, "def f(x):\n    while x:\n        x -= 1\n    return x\n"])
    assert per_sample[0] == 0.0
    assert per_sample[1] > 0.0
    assert abs(score - sum(per_sample) / 2) < 1e-9
    assert scorer.parse_failures == 0


def test_scorer_counts_unparseable_sample_as_max_divergence():
    main = "def f(x):\n    return x + 1\n"
    scorer = ASTScorer()
    score, per_sample = scorer.evaluate(main, ["@@@ not python @@@"])
    assert per_sample == [1.0]
    assert score == 1.0
    assert scorer.parse_failures == 1


def test_scorer_unparseable_main_scores_max_and_counts_once():
    scorer = ASTScorer()
    score, per_sample = scorer.evaluate("@@@ bad main @@@", ["def f(): return 1", "def g(): return 2"])
    assert score == 1.0
    assert per_sample == [1.0, 1.0]
    assert scorer.parse_failures == 1


def test_scorer_empty_samples_scores_zero():
    scorer = ASTScorer()
    assert scorer.score("def f(): return 1", []) == 0.0


def test_scorer_blank_sample_is_max_divergence_and_counted():
    # A blank/comment-only sample is a non-answer: treat as a parse failure.
    scorer = ASTScorer()
    score, per_sample = scorer.evaluate("def f(): return 1", ["", "# nothing\n"])
    assert per_sample == [1.0, 1.0]
    assert score == 1.0
    assert scorer.parse_failures == 2


def test_scorer_score_wraps_evaluate():
    main = "def f(x):\n    return x + 1\n"
    scorer = ASTScorer()
    assert scorer.score(main, [main]) == 0.0
