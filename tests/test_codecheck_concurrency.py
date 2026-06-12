import time

from codecheck.generation.concurrency import map_staggered, STAGGER_DELAY


def test_preserves_input_order():
    assert map_staggered(lambda x: x * 2, [1, 2, 3], delay=0.0) == [2, 4, 6]


def test_empty_items_returns_empty():
    assert map_staggered(lambda x: x, [], delay=0.0) == []


def test_default_stagger_delay_is_point_one():
    assert STAGGER_DELAY == 0.1


def test_staggers_call_starts_by_delay():
    # item i fires ~i*delay after start, so n items take at least (n-1)*delay wall-clock.
    delay = 0.05
    n = 4
    started = time.monotonic()
    map_staggered(lambda x: x, list(range(n)), delay=delay)
    elapsed = time.monotonic() - started
    assert elapsed >= (n - 1) * delay * 0.9


def test_zero_delay_does_not_stagger():
    started = time.monotonic()
    map_staggered(lambda x: x, list(range(10)), delay=0.0)
    assert time.monotonic() - started < 0.1
