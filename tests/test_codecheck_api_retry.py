import json
from types import SimpleNamespace

import pytest

import codecheck.generation.api_retry as api_retry
from codecheck.generation.api_retry import APIRetriesExhausted, chat_with_retries


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


class ResponseClient:
    """Returns one configurable chat completion (finish_reason / content / usage)."""

    def __init__(self, content="ok", finish_reason="stop", completion_tokens=12):
        usage = SimpleNamespace(completion_tokens=completion_tokens)
        self._resp = SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )],
            usage=usage,
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: self._resp))


def test_successful_call_logs_return_value_clues_at_debug(caplog):
    c = ResponseClient(content="def f(): return 1", finish_reason="stop", completion_tokens=7)
    with caplog.at_level("DEBUG", logger="codecheck.api"):
        chat_with_retries(c, model="m", messages=[], temperature=0.0, think=False)
    debug = [r.message for r in caplog.records if r.levelname == "DEBUG"]
    assert any("finish=stop" in m and "completion_tokens=7" in m for m in debug)


def test_truncated_finish_reason_warns(caplog):
    c = ResponseClient(content="def f(): retur", finish_reason="length")
    with caplog.at_level("WARNING", logger="codecheck.api"):
        chat_with_retries(c, model="m", messages=[], temperature=0.0, think=False)
    assert any("truncated" in r.message and "length" in r.message
               for r in caplog.records if r.levelname == "WARNING")


def test_empty_content_warns(caplog):
    c = ResponseClient(content="", finish_reason="stop")
    with caplog.at_level("WARNING", logger="codecheck.api"):
        chat_with_retries(c, model="m", messages=[], temperature=0.0, think=False)
    assert any("empty content" in r.message
               for r in caplog.records if r.levelname == "WARNING")


def test_clean_success_emits_no_warning(caplog):
    c = ResponseClient(content="def f(): return 1", finish_reason="stop")
    with caplog.at_level("WARNING", logger="codecheck.api"):
        chat_with_retries(c, model="m", messages=[], temperature=0.0, think=False)
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_abandoned_call_runs_on_a_daemon_thread():
    # A wall-clock-abandoned call leaves its underlying thread still blocked in the SDK.
    # That thread MUST be a daemon, or the interpreter force-joins it at exit and the
    # program hangs after the run finishes (the real bug this guards).
    import threading
    c = StarvingClient(slow_times=1, block_seconds=5.0)
    chat_with_retries(c, model="m", messages=[], temperature=0.0, think=False,
                      attempts=2, call_timeout=0.05)
    lingering = [t for t in threading.enumerate() if t.name.startswith("chat-") and t.is_alive()]
    assert lingering                      # the starved call's thread is still blocking
    assert all(t.daemon for t in lingering)
