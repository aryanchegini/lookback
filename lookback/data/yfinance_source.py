from datetime import datetime

import pandas as pd
import yfinance as yf

from lookback.data.source import DataSource
from lookback.core.exceptions import SymbolNotFoundError, DataError


class YFinanceSource(DataSource):
    """Fetches bars from Yahoo Finance via the yfinance library."""

    def fetch_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        # 1. download, wrapping any failure as a DataError
        # 2. if the result is empty -> SymbolNotFoundError
        # 3. normalise and return
        try:
            df = yf.download(symbol, start=start, end=end)
        except Exception as e:
              raise DataError(f"Failed to download {symbol}") from e
        
        if df.empty:
            raise SymbolNotFoundError(f"No data returned for {symbol}")
        
        # normalise
        df = df.droplevel("Ticker", axis=1)
        df.columns = df.columns.str.lower()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
            
