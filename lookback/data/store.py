from datetime import datetime, timedelta

from lookback.data.cache import ParquetCache
from lookback.data.source import DataSource
from lookback.exceptions import InsufficientDataError


class DataStore:
    """Point-in-time bar access, optionally backed by a parquet cache.

    get_bars is read-through: serve from cache, and on a miss fetch from the
    source and populate the cache. The as-of slice runs AFTER the read, so
    the point-in-time guarantee holds whether data came from cache or source.
    """

    def __init__(self, data_source: DataSource, cache: ParquetCache | None = None):
        self.data_source = data_source
        self.cache = cache

    def get_bars(self, symbol: str, as_of: datetime, start: datetime | None = None):
        if start is None:
            start = as_of - timedelta(days=365.0)

        df = None
        if self.cache is not None:
            df = self.cache.read(symbol, start, as_of)

        if df is None or df.empty:
            df = self.data_source.fetch_bars(symbol=symbol, start=start, end=as_of)
            if self.cache is not None and not df.empty:
                self.cache.write(symbol, df)

        # Point-in-time guard: never return anything dated after as_of.
        df = df[df.index <= as_of]
        if df.empty:
            raise InsufficientDataError(f"no data for {symbol} as of {as_of:%Y-%m-%d}")
        return df
