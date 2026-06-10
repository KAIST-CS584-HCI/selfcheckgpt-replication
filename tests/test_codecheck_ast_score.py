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


from codecheck.ast_score import ast_to_tree, ted_dissimilarity


def test_ted_identical_structure_is_zero():
    t = ast_to_tree("def f(x):\n    return x + 1\n")
    assert ted_dissimilarity(t, t) == 0.0


def test_ted_is_invariant_to_variable_renaming():
    a = ast_to_tree("def f(x):\n    return x + 1\n")
    b = ast_to_tree("def f(y):\n    return y + 1\n")
    assert ted_dissimilarity(a, b) == 0.0


def test_ted_is_invariant_to_literal_values():
    a = ast_to_tree("def f(x):\n    return x + 1\n")
    b = ast_to_tree("def f(x):\n    return x + 99\n")
    assert ted_dissimilarity(a, b) == 0.0


def test_ted_structural_difference_is_positive_and_bounded():
    a = ast_to_tree("def f(x):\n    return x + 1\n")
    b = ast_to_tree("def f(x):\n    for i in range(x):\n        print(i)\n")
    d = ted_dissimilarity(a, b)
    assert 0.0 < d <= 1.0


def test_ted_sees_nesting_that_jaccard_misses():
    # Same node-type multiset (left- vs right-associative), different tree shape.
    # Jaccard scores them identical; TED, which compares structure, must not.
    a_code = "x = (a + b) + c\n"
    b_code = "x = a + (b + c)\n"
    assert ast_dissimilarity(ast_fingerprint(a_code), ast_fingerprint(b_code)) == 0.0
    assert ted_dissimilarity(ast_to_tree(a_code), ast_to_tree(b_code)) > 0.0


def test_ast_to_tree_returns_none_on_unparseable_or_bodyless():
    assert ast_to_tree("@@@ not python @@@") is None
    assert ast_to_tree("") is None
    assert ast_to_tree("# only a comment\n") is None
    assert ast_to_tree(None) is None


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


def test_scorer_defaults_to_ted_metric():
    assert ASTScorer().metric == "ted"


def test_scorer_rejects_unknown_metric():
    import pytest
    with pytest.raises(ValueError):
        ASTScorer(metric="bogus")


def test_scorer_ted_sees_nesting_that_jaccard_metric_misses():
    main = "x = (a + b) + c\n"
    sample = "x = a + (b + c)\n"
    assert ASTScorer(metric="jaccard").score(main, [sample]) == 0.0
    assert ASTScorer(metric="ted").score(main, [sample]) > 0.0


def test_scorer_jaccard_metric_still_available():
    main = "def f(x):\n    return x + 1\n"
    scorer = ASTScorer(metric="jaccard")
    assert scorer.score(main, [main]) == 0.0
    assert scorer.metric == "jaccard"


def test_scorer_ted_counts_parse_failures():
    scorer = ASTScorer(metric="ted")
    score, per_sample = scorer.evaluate("def f(): return 1", ["@@@ bad @@@", "def g(): return 2"])
    assert per_sample[0] == 1.0
    assert scorer.parse_failures == 1
