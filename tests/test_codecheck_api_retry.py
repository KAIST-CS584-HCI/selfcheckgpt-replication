import json
from types import SimpleNamespace

import pytest

import codecheck.api_retry as api_retry
from codecheck.api_retry import APIRetriesExhausted, chat_with_retries


class FlakyClient:
    """Raises `exc` for the first `fail_times` calls, then returns a normal response."""

    def __init__(self, fail_times, exc=None):
        self.fail_times = fail_times
        self.exc = exc or json.JSONDecodeError("Expecting value", "", 0)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(api_retry.time, "sleep", lambda *_: None)


def test_retries_transient_then_succeeds():
    c = FlakyClient(fail_times=2)
    resp = chat_with_retries(c, model="m", messages=[], temperature=0.0, think=False, attempts=4)
    assert resp.choices[0].message.content == "ok"
    assert c.calls == 3


def test_exhausted_raises_apiretriesexhausted():
    c = FlakyClient(fail_times=99)
    with pytest.raises(APIRetriesExhausted):
        chat_with_retries(c, model="m", messages=[], temperature=0.0, think=False, attempts=3)
    assert c.calls == 3


def test_non_transient_error_is_not_retried():
    c = FlakyClient(fail_times=99, exc=ValueError("config problem"))
    with pytest.raises(ValueError):
        chat_with_retries(c, model="m", messages=[], temperature=0.0, think=False, attempts=4)
    assert c.calls == 1


def test_retry_is_logged(caplog):
    c = FlakyClient(fail_times=1)
    with caplog.at_level("WARNING", logger="codecheck.api"):
        chat_with_retries(c, model="m", messages=[], temperature=0.0, think=False, attempts=3)
    assert any("transient API error" in r.message for r in caplog.records)
