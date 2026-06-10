from __future__ import annotations
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from openai import APIConnectionError, InternalServerError, RateLimitError

logger = logging.getLogger("codecheck.api")

# Transient failures worth retrying. JSONDecodeError covers a malformed/truncated 200 body
# (an OpenRouter gateway/HTML error page parsed as JSON) — the SDK does NOT retry that.
# Auth / bad-request / not-found errors are persistent config problems and are NOT retried;
# they propagate immediately so a misconfigured run fails fast instead of looping.
_TRANSIENT = (APIConnectionError, InternalServerError, RateLimitError, json.JSONDecodeError)


class APIRetriesExhausted(Exception):
    """Raised when transient API failures persisted across every retry attempt."""


def chat_with_retries(client, *, model, messages, temperature, think,
                      attempts: int = 4, base_delay: float = 0.8, call_timeout: float = 60.0):
    """One chat completion, retried on transient failures with exponential backoff.

    Each attempt is bounded by `call_timeout` seconds of WALL-CLOCK time. The SDK's own
    timeout keys off the gap *between* received bytes, so under load a gateway that
    trickles keepalive bytes can wedge a request indefinitely without ever raising. The
    watchdog here runs the call on a worker thread and abandons it after `call_timeout`,
    treating the stall as a transient failure and retrying. The abandoned thread is left
    to unwind on the SDK's own timeout; `max_workers=attempts` keeps a stalled thread from
    blocking the next attempt.

    Returns the SDK response. Raises APIRetriesExhausted if every attempt hit a transient
    error or stalled; re-raises non-transient errors (e.g. AuthenticationError) immediately.
    """
    last: Exception | None = None
    executor = ThreadPoolExecutor(max_workers=attempts, thread_name_prefix="chat")
    try:
        for attempt in range(attempts):
            future = executor.submit(
                client.chat.completions.create,
                model=model,
                messages=messages,
                temperature=temperature,
                extra_body={"reasoning": {"enabled": think}},
            )
            try:
                return future.result(timeout=call_timeout)
            except FuturesTimeout as exc:
                last = exc
                future.cancel()
                logger.warning("API call exceeded %.0fs wall-clock; abandoning and retrying %d/%d",
                               call_timeout, attempt + 1, attempts - 1)
            except _TRANSIENT as exc:
                last = exc
                if attempt < attempts - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning("transient API error %s; retry %d/%d after %.1fs",
                                   type(exc).__name__, attempt + 1, attempts - 1, delay)
                    time.sleep(delay)
        logger.error("API call failed after %d attempts (%s)", attempts, type(last).__name__)
        raise APIRetriesExhausted(repr(last)) from last
    finally:
        executor.shutdown(wait=False)
