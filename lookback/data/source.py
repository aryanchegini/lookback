from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class DataSource(ABC):
    """A source of historical OHLCV bars."""

    @abstractmethod
    def fetch_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch raw bars for `symbol` in [start, end]. Implementations
          normalise to: lowercase OHLCV columns, a datetime index, no tz."""
        ...

