from datetime import datetime

import numpy as np
import pandas as pd

from lookback.data.cache import ParquetCache
from lookback.data.source import DataSource
from lookback.data.store import DataStore


def _ohlcv(dates):
    idx = pd.to_datetime(dates)
    n = len(idx)
    close = np.arange(1.0, n + 1.0)
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": close * 10}, index=idx)


class _CountingSource(DataSource):
    """Returns a fixed OHLCV frame and counts how many times it was called."""

    def __init__(self, frame):
        self.frame = frame
        self.calls = 0

    def fetch_bars(self, symbol, start, end):
        self.calls += 1
        return self.frame


# ---------- cache round-trip / pushdown ----------

def test_write_then_read_roundtrip(tmp_path):
    cache = ParquetCache(tmp_path)
    frame = _ohlcv(["2024-01-01", "2024-01-02", "2024-01-03"])
    cache.write("AAA", frame)

    out = cache.read("AAA", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03"))
    # cache returns the index named "date"; values must be identical.
    pd.testing.assert_frame_equal(out, frame.rename_axis("date"), check_freq=False)


def test_read_prunes_by_symbol(tmp_path):
    cache = ParquetCache(tmp_path)
    cache.write("AAA", _ohlcv(["2024-01-01", "2024-01-02"]))
    cache.write("BBB", _ohlcv(["2024-01-01", "2024-01-02"]))

    out = cache.read("AAA", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
    assert len(out) == 2  # only AAA's rows, BBB pruned by the symbol filter


def test_read_filters_date_range(tmp_path):
    cache = ParquetCache(tmp_path)
    cache.write("AAA", _ohlcv(["2024-01-01", "2024-06-01", "2024-12-01"]))

    out = cache.read("AAA", pd.Timestamp("2024-05-01"), pd.Timestamp("2024-07-01"))
    assert list(out.index) == [pd.Timestamp("2024-06-01")]


def test_partition_layout_on_disk(tmp_path):
    cache = ParquetCache(tmp_path)
    cache.write("AAA", _ohlcv(["2023-12-31", "2024-01-01"]))
    # Hive dirs: symbol=AAA/year=2023, symbol=AAA/year=2024
    assert (tmp_path / "symbol=AAA" / "year=2023").exists()
    assert (tmp_path / "symbol=AAA" / "year=2024").exists()


def test_read_missing_returns_empty(tmp_path):
    cache = ParquetCache(tmp_path)
    out = cache.read("NONE", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
    assert out.empty


# ---------- DataStore read-through ----------

def test_read_through_populates_cache_once(tmp_path):
    frame = _ohlcv(["2024-01-01", "2024-01-02", "2024-01-03"])
    src = _CountingSource(frame)
    store = DataStore(src, ParquetCache(tmp_path))

    store.get_bars("AAA", as_of=datetime(2024, 1, 3))
    store.get_bars("AAA", as_of=datetime(2024, 1, 3))
    assert src.calls == 1  # second call served from cache


def test_as_of_holds_through_cache(tmp_path):
    frame = _ohlcv(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    src = _CountingSource(frame)
    store = DataStore(src, ParquetCache(tmp_path))

    # First call fetches + caches the full range but slices to as_of.
    early = store.get_bars("AAA", as_of=datetime(2024, 1, 3))
    assert len(early) == 3
    # Later as_of served from cache still respects the point-in-time bound.
    later = store.get_bars("AAA", as_of=datetime(2024, 1, 5))
    assert len(later) == 5
    assert src.calls == 1
