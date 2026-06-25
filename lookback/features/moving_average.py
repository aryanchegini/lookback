import pandas as pd

from lookback.features.base import Feature


class MovingAverage(Feature):
    def __init__(self, window: int = 50):
        self.window = window

    @property
    def name(self) -> str:
        return f"sma_{self.window}"

    def compute(self, prices: pd.DataFrame) -> pd.Series:
        return prices["close"].rolling(self.window).mean()
