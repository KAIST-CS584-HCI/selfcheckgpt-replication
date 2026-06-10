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


def test_identical_structure_has_zero_dissimilarity():
    a = ast_fingerprint("def f(x):\n    return x + 1\n")
    assert ast_dissimilarity(a, a) == 0.0


def test_different_structure_has_positive_dissimilarity():
    a = ast_fingerprint("def f(x):\n    return x + 1\n")
    b = ast_fingerprint("def f(x):\n    for i in range(x):\n        print(i)\n")
    d = ast_dissimilarity(a, b)
    assert 0.0 < d <= 1.0
