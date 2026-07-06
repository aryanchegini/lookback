from lookback.utils.decorators import memoize, timed


def test_timed_records_duration_and_preserves_identity():
    @timed
    def add(a, b):
        """adds."""
        return a + b

    assert add(2, 3) == 5
    assert add.last_seconds is not None and add.last_seconds >= 0
    assert add.__name__ == "add"        # functools.wraps kept identity
    assert add.__doc__ == "adds."


def test_memoize_caches_and_counts():
    @memoize
    def square(x):
        return x * x

    assert square(4) == 16
    assert square(4) == 16   # served from cache
    assert square(5) == 25
    assert square.calls == 2  # ran twice (x=4 once, x=5 once), not three times


def test_memoize_handles_kwargs():
    @memoize
    def combine(a, b=0):
        return a + b

    assert combine(1, b=2) == 3
    assert combine(1, b=2) == 3
    assert combine(1, b=3) == 4
    assert combine.calls == 2
