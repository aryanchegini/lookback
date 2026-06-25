import numpy as np
import pandas as pd

from lookback.features.base import Feature


class RollingVolatility(Feature):
    def __init__(self, window: int = 20, annualise: bool = True):
        self.window = window
        self.annualise = annualise

    @property
    def name(self) -> str:
        return f"rolling_vol_{self.window}"

    def compute(self, prices: pd.DataFrame) -> pd.Series:
        log_returns = np.log(prices["close"]).diff()
        vol = log_returns.rolling(self.window).std()
        if self.annualise:
            vol = vol * np.sqrt(252)
        return vol
