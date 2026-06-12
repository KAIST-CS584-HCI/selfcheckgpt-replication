from codecheck.score.prompt import build_judge_prompt, parse_judgment


def test_build_judge_prompt_includes_both_codes():
    p = build_judge_prompt("def f(): return 1", "def f(): return 2")
    assert "def f(): return 1" in p
    assert "def f(): return 2" in p
    assert "Yes" in p and "No" in p


def test_humaneval_template_is_divergence_seeking():
    from codecheck.score.prompt import HUMANEVAL_JUDGE_TEMPLATE
    p = build_judge_prompt("def f(): return 1", "def f(): return 2", HUMANEVAL_JUDGE_TEMPLATE)
    assert "def f(): return 1" in p and "def f(): return 2" in p
    assert "Yes" in p and "No" in p
    # distinctive edge-case divergence framing (vs the default consistency prompt)
    assert "every input" in p.lower() and "edge cases" in p.lower()


def test_codehalu_template_is_program_oriented_and_divergence_seeking():
    from codecheck.score.prompt import CODEHALU_JUDGE_TEMPLATE
    p = build_judge_prompt("print(1)", "print(2)", CODEHALU_JUDGE_TEMPLATE)
    assert "print(1)" in p and "print(2)" in p
    assert "Yes" in p and "No" in p
    assert "stdin" in p.lower() and "stdout" in p.lower()
    assert "every valid stdin" in p.lower() and "edge cases" in p.lower()


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


import threading
from types import SimpleNamespace
from codecheck.score.prompt import PromptJudge


class FakeJudgeClient:
    """Returns a queued answer per call, in call order."""
    def __init__(self, answers):
        self._answers = list(answers)
        self.calls = []
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        with self._lock:
            idx = len(self.calls)
            self.calls.append(kwargs)
        content = self._answers[idx]
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_judge_score_is_mean_inconsistency():
    client = FakeJudgeClient(["Yes.", "No.", "Yes."])  # 0.0, 1.0, 0.0 -> mean 1/3
    judge = PromptJudge(client, model="m")
    score = judge.score("def f(): return 1", ["a", "b", "c"])
    assert abs(score - (1.0 / 3.0)) < 1e-9
    assert judge.parse_failures == 0


def test_judge_uses_custom_template_and_keeps_mapping():
    from codecheck.score.prompt import HUMANEVAL_JUDGE_TEMPLATE
    client = FakeJudgeClient(["No."])  # No -> 1.0 under any template
    judge = PromptJudge(client, model="m", template=HUMANEVAL_JUDGE_TEMPLATE)
    assert judge.score("def f(): return 1", ["x"]) == 1.0
    # the custom template's distinctive phrasing reached the API call
    assert "edge cases" in client.calls[0]["messages"][0]["content"].lower()


def test_judge_evaluate_returns_raw_responses():
    client = FakeJudgeClient(["Yes.", "No.", "Yes."])
    judge = PromptJudge(client, model="m")
    score, raws = judge.evaluate("def f(): return 1", ["a", "b", "c"])
    assert abs(score - (1.0 / 3.0)) < 1e-9
    # ex.map fixes result order to sample order, but the fake hands out answers in
    # thread-scheduling order, so assert the multiset rather than exact positions.
    assert sorted(raws) == ["No.", "Yes.", "Yes."]


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


def test_judge_think_enables_reasoning():
    client = FakeJudgeClient(["Yes."])
    PromptJudge(client, model="m", think=True).score("main", ["a"])
    assert client.calls[0]["extra_body"] == {"reasoning": {"enabled": True}}


def test_judge_degrades_on_persistent_api_failure(monkeypatch):
    # A judge call that keeps failing (transient API error across all retries) must not
    # crash the run: the sample degrades to a parse failure (N/A) and the run continues.
    import json
    import codecheck.generation.api_retry as api_retry
    monkeypatch.setattr(api_retry.time, "sleep", lambda *_: None)

    class DeadClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._c))
        def _c(self, **kwargs):
            raise json.JSONDecodeError("Expecting value", "", 0)

    judge = PromptJudge(DeadClient(), model="m")
    score = judge.score("main", ["a", "b"])
    assert score == 0.5               # both unusable -> N/A
    assert judge.parse_failures == 2
