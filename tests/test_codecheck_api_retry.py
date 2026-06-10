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


import threading


class StarvingClient:
    """Blocks longer than the wall-clock call_timeout on the first `slow_times`
    calls (simulating a request OpenRouter accepts but never finishes under load),
    then returns a normal response. Blocks via Event.wait so the autouse sleep patch
    (which targets time.sleep) does not neutralize the simulated stall."""

    def __init__(self, slow_times, block_seconds=2.0):
        self.slow_times = slow_times
        self.block_seconds = block_seconds
        self.calls = 0
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        with self._lock:
            self.calls += 1
            n = self.calls
        if n <= self.slow_times:
            threading.Event().wait(self.block_seconds)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


def test_wall_clock_timeout_abandons_starved_call_and_retries():
    # Every attempt starves past the call_timeout -> each is abandoned and retried,
    # finally exhausting. Without a wall-clock cap this would block on the first call.
    c = StarvingClient(slow_times=99, block_seconds=2.0)
    with pytest.raises(APIRetriesExhausted):
        chat_with_retries(c, model="m", messages=[], temperature=0.0, think=False,
                          attempts=3, call_timeout=0.05)
    assert c.calls == 3


def test_retry_recovers_when_a_later_call_returns_fast():
    # First call starves; the retry returns immediately.
    c = StarvingClient(slow_times=1, block_seconds=2.0)
    resp = chat_with_retries(c, model="m", messages=[], temperature=0.0, think=False,
                             attempts=3, call_timeout=0.05)
    assert resp.choices[0].message.content == "ok"
    assert c.calls >= 2


def test_wall_clock_timeout_is_logged(caplog):
    c = StarvingClient(slow_times=1, block_seconds=2.0)
    with caplog.at_level("WARNING", logger="codecheck.api"):
        chat_with_retries(c, model="m", messages=[], temperature=0.0, think=False,
                          attempts=3, call_timeout=0.05)
    assert any("wall-clock" in r.message for r in caplog.records)
