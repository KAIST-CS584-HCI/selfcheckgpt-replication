from __future__ import annotations
import json
import logging
import threading
import time

from openai import APIConnectionError, InternalServerError, RateLimitError

logger = logging.getLogger("codecheck.api")

# Transient failures worth retrying. JSONDecodeError covers a malformed/truncated 200 body
# (an OpenRouter gateway/HTML error page parsed as JSON) — the SDK does NOT retry that.
# Auth / bad-request / not-found errors are persistent config problems and are NOT retried;
# they propagate immediately so a misconfigured run fails fast instead of looping.
_TRANSIENT = (APIConnectionError, InternalServerError, RateLimitError, json.JSONDecodeError)


class APIRetriesExhausted(Exception):
    """Raised when transient API failures persisted across every retry attempt."""


def _log_response(resp, elapsed: float) -> None:
    """Surface the return-value clues from a successful completion.

    Per-call detail (latency, finish_reason, content size, completion tokens) goes to
    DEBUG so a normal run stays quiet; the two anomalies worth seeing by default —
    a truncated response (`finish_reason` other than `stop`) and empty content — warn.
    """
    choice = resp.choices[0] if getattr(resp, "choices", None) else None
    finish = getattr(choice, "finish_reason", None)
    message = getattr(choice, "message", None)
    content = (getattr(message, "content", None) or "") if message else ""
    usage = getattr(resp, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

    logger.debug("API ok in %.1fs: finish=%s chars=%d completion_tokens=%s",
                 elapsed, finish, len(content), completion_tokens)
    if finish not in (None, "stop"):
        logger.warning("API finish_reason=%s after %.1fs: output likely truncated/incomplete",
                       finish, elapsed)
    if not content:
        logger.warning("API returned empty content after %.1fs (finish=%s)", elapsed, finish)


def chat_with_retries(client, *, model, messages, temperature, think,
                      attempts: int = 4, base_delay: float = 0.8, call_timeout: float = 60.0):
    """One chat completion, retried on transient failures with exponential backoff.

    Each attempt is bounded by `call_timeout` seconds of WALL-CLOCK time. The SDK's own
    timeout keys off the gap *between* received bytes, so under load a gateway that
    trickles keepalive bytes can wedge a request indefinitely without ever raising. Each
    attempt runs the call on a **daemon** thread and abandons it after `call_timeout`,
    treating the stall as a transient failure and retrying. The abandoned daemon thread
    unwinds on the SDK's own timeout — and, being a daemon, never keeps the interpreter
    alive at exit (a `ThreadPoolExecutor` would force-join it, hanging the program after
    the run finished).

    Returns the SDK response. Raises APIRetriesExhausted if every attempt hit a transient
    error or stalled; re-raises non-transient errors (e.g. AuthenticationError) immediately.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        box: dict = {}
        done = threading.Event()

        def _call() -> None:
            try:
                box["resp"] = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    extra_body={"reasoning": {"enabled": think}},
                )
            except Exception as exc:  # noqa: BLE001 — inspected by the waiting thread
                box["exc"] = exc
            finally:
                done.set()

        started = time.monotonic()
        threading.Thread(target=_call, name=f"chat-{attempt}", daemon=True).start()
        if not done.wait(call_timeout):
            last = TimeoutError()
            logger.warning("API call exceeded %.0fs wall-clock; abandoning and retrying %d/%d",
                           call_timeout, attempt + 1, attempts - 1)
            continue
        exc = box.get("exc")
        if exc is None:
            _log_response(box["resp"], time.monotonic() - started)
            return box["resp"]
        if isinstance(exc, _TRANSIENT):
            last = exc
            if attempt < attempts - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning("transient API error %s; retry %d/%d after %.1fs",
                               type(exc).__name__, attempt + 1, attempts - 1, delay)
                time.sleep(delay)
            continue
        raise exc  # non-transient (e.g. AuthenticationError): surface immediately

    logger.error("API call failed after %d attempts (%s)", attempts, type(last).__name__)
    raise APIRetriesExhausted(repr(last)) from last
