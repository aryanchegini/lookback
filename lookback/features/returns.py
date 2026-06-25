import numpy as np
import pandas as pd

from lookback.features.base import Feature


class SimpleReturns(Feature):
    @property
    def name(self) -> str:
        return "simple_returns"

    def compute(self, prices: pd.DataFrame) -> pd.Series:
        return prices["close"].pct_change()


class LogReturns(Feature):
    @property
    def name(self) -> str:
        return "log_returns"

    def compute(self, prices: pd.DataFrame) -> pd.Series:
        return np.log(prices["close"]).diff()
