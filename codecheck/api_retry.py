from __future__ import annotations
import json
import logging
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


def chat_with_retries(client, *, model, messages, temperature, think,
                      attempts: int = 4, base_delay: float = 0.8):
    """One chat completion, retried on transient failures with exponential backoff.

    Returns the SDK response. Raises APIRetriesExhausted if every attempt hit a transient
    error; re-raises non-transient errors (e.g. AuthenticationError) on the first try.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                extra_body={"reasoning": {"enabled": think}},
            )
        except _TRANSIENT as exc:
            last = exc
            if attempt < attempts - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning("transient API error %s; retry %d/%d after %.1fs",
                               type(exc).__name__, attempt + 1, attempts - 1, delay)
                time.sleep(delay)
    logger.error("API call failed after %d attempts (%s)", attempts, type(last).__name__)
    raise APIRetriesExhausted(repr(last)) from last
