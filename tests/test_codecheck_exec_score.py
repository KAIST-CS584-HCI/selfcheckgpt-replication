from codecheck.exec_score import exec_inconsistency

MAIN = [("value", 1), ("value", 2), ("value", 3)]


def test_all_samples_agree_is_zero():
    assert exec_inconsistency(MAIN, [MAIN, MAIN]) == 0.0


def test_all_samples_disagree_is_one():
    other = [("value", 9), ("value", 9), ("value", 9)]
    assert exec_inconsistency(MAIN, [other, other]) == 1.0


def test_partial_agreement():
    half = [("value", 1), ("value", 2), ("value", 9)]  # 2/3 match
    assert exec_inconsistency(MAIN, [half]) == 1.0 - (2 / 3)


def test_no_samples_is_zero():
    assert exec_inconsistency(MAIN, []) == 0.0
