import pandas as pd

from lookback.features.base import Feature


class Momentum(Feature):
    def __init__(self, window: int = 20):
        self.window = window

    @property
    def name(self) -> str:
        return f"momentum_{self.window}"

    def compute(self, prices: pd.DataFrame) -> pd.Series:
        return prices["close"].pct_change(self.window)
