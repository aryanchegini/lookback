from datetime import datetime
from dataclasses import FrozenInstanceError

import pytest

from lookback.core.instruments import Bar


def _bar():
    return Bar(datetime(2024, 1, 2), "AAPL", 1.0, 2.0, 0.5, 1.5, 100.0)


def test_bar_equality_by_value():
    # two Bars with identical fields should be ==
    bar1 = _bar()
    bar2 = _bar()
    
    assert bar1 == bar2


def test_bar_is_frozen():
    # assigning to a field raises FrozenInstanceError
    # use: with pytest.raises(FrozenInstanceError): ...
    bar = _bar()
    with pytest.raises(FrozenInstanceError):
        bar.close = 10.0
    

def test_bar_uses_slots():
    # no per-instance __dict__, and __slots__ lists the fields
    bar = _bar()
    assert not hasattr(bar, "__dict__")
    assert bar.__slots__ == ("timestamp", "symbol", "open", "high", "low", "close", "volume")
    
