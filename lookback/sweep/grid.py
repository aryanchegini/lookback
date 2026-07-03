import itertools
from collections.abc import Iterator


def param_grid(**axes) -> Iterator[dict]:
    """Lazily yield one config dict per combination of the given axes.

    A generator on purpose: the full grid is never materialised, so the
    same source feeds a tiny loop or a huge parallel sweep unchanged.

        param_grid(fast=[10, 20], slow=[50, 100])
        -> {'fast': 10, 'slow': 50}, {'fast': 10, 'slow': 100}, ...
    """
    keys = list(axes)
    for combo in itertools.product(*axes.values()):
        yield dict(zip(keys, combo))
