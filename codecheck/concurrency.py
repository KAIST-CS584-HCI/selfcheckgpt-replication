from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Seconds to wait between consecutive concurrent API request launches. A small stagger
# spreads a burst of N+1 simultaneous calls over time so we don't trip the provider's
# rate limit, while still running them concurrently.
STAGGER_DELAY = 0.1


def map_staggered(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    delay: float = STAGGER_DELAY,
    max_workers: int | None = None,
) -> list[R]:
    """Run `fn` over `items` concurrently, staggering each call's start: item i waits
    `i * delay` seconds before firing. Preserves input order, like
    `ThreadPoolExecutor.map`. With `delay=0` it is a plain concurrent map."""
    items = list(items)
    if not items:
        return []

    def run(indexed: tuple[int, T]) -> R:
        i, item = indexed
        if i and delay:
            time.sleep(i * delay)
        return fn(item)

    with ThreadPoolExecutor(max_workers=max_workers or len(items)) as ex:
        return list(ex.map(run, enumerate(items)))
