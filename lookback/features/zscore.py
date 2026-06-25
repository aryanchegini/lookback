import pandas as pd

from lookback.features.base import Feature


class ZScore(Feature):
    def __init__(self, window: int = 60):
        self.window = window

    @property
    def name(self) -> str:
        return f"zscore_{self.window}"

    def compute(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        window = close.rolling(self.window)
        return (close - window.mean()) / window.std()
