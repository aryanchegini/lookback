from datetime import datetime, timedelta

import pandas as pd

from lookback.data.source import DataSource
from lookback.core.exceptions import InsufficientDataError


class DataStore:
    def __init__(self, data_source: DataSource):
        self.data_source = data_source
        
    def get_bars(self, symbol: str, as_of: datetime, start: datetime | None = None):
        if start is None:
            start = as_of - timedelta(days=365.0)
        df = self.data_source.fetch_bars(symbol=symbol, start=start, end=as_of)
        df = df[df.index <= as_of]
        if df.empty:
            raise InsufficientDataError(f"no data for {symbol} as of {as_of:%Y-%m-%d}")
        return df
    
        
