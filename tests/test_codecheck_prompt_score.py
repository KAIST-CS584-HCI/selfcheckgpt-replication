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


from types import SimpleNamespace
from codecheck.prompt_score import PromptJudge


class FakeJudgeClient:
    """Returns a queued answer per call, in call order."""
    def __init__(self, answers):
        self._answers = list(answers)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._answers[len(self.calls) - 1]
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_judge_score_is_mean_inconsistency():
    client = FakeJudgeClient(["Yes.", "No.", "Yes."])  # 0.0, 1.0, 0.0 -> mean 1/3
    judge = PromptJudge(client, model="m")
    score = judge.score("def f(): return 1", ["a", "b", "c"])
    assert abs(score - (1.0 / 3.0)) < 1e-9
    assert judge.parse_failures == 0


def test_judge_counts_parse_failures():
    client = FakeJudgeClient(["Yes.", "uhh dunno"])     # second is unparseable
    judge = PromptJudge(client, model="m")
    judge.score("main", ["a", "b"])
    assert judge.parse_failures == 1


def test_judge_empty_samples_scores_zero():
    client = FakeJudgeClient([])
    judge = PromptJudge(client, model="m")
    assert judge.score("main", []) == 0.0


def test_judge_disables_reasoning_by_default():
    client = FakeJudgeClient(["Yes."])
    PromptJudge(client, model="m").score("main", ["a"])
    assert client.calls[0]["extra_body"] == {"reasoning": {"enabled": False}}
    assert client.calls[0]["temperature"] == 0.0
