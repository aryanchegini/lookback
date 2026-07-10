from datetime import datetime

import pandas as pd
import pytest

from lookback.data.source import DataSource
from lookback.data.store import DataStore
from lookback.exceptions import InsufficientDataError


class _FakeSource(DataSource):
    """Returns a fixed 5-day frame regardless of args - including 'future' bars."""

    def fetch_bars(self, symbol, start, end):
        idx = pd.to_datetime([
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        ])
        return pd.DataFrame({"close": [1, 2, 3, 4, 5]}, index=idx)


def _store():
    return DataStore(_FakeSource())


def test_no_lookahead():
    # source has bars up to Jan 5, but as_of is Jan 3 -> Jan 4 & 5 must be gone
    df = _store().get_bars("X", as_of=datetime(2024, 1, 3))
    assert (df.index <= datetime(2024, 1, 3)).all()
    assert len(df) == 3


def test_as_of_is_inclusive():
    # the as_of date itself is included (boundary is <=, not <)
    df = _store().get_bars("X", as_of=datetime(2024, 1, 3))
    assert pd.Timestamp("2024-01-03") in df.index


def test_no_data_raises():
    # as_of before any available bar -> nothing survives the slice
    with pytest.raises(InsufficientDataError):
        _store().get_bars("X", as_of=datetime(2023, 1, 1))
