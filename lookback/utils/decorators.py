import functools
import time
from collections.abc import Callable


def timed(func: Callable) -> Callable:
    """Measure wall-clock time of each call; expose it as wrapper.last_seconds.

    functools.wraps copies the wrapped function's identity (name, docstring)
    onto the wrapper so it still looks like the original.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        wrapper.last_seconds = time.perf_counter() - start
        return result

    wrapper.last_seconds = None
    return wrapper


def memoize(func: Callable) -> Callable:
    """Cache results by (args, kwargs), so repeated calls skip recomputation.

    Exposes wrapper.cache (the dict) and wrapper.calls (how many times the
    underlying function actually ran).
    """
    cache: dict = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
            wrapper.calls += 1
        return cache[key]

    wrapper.cache = cache
    wrapper.calls = 0
    return wrapper
