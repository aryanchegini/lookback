import numpy as np
import pandas as pd

from lookback.features.returns import SimpleReturns
from lookback.features.volatility import RollingVolatility
from lookback.strategies.base import Strategy


class VolBreakout(Strategy):
    """Chase outsized moves: go with the direction of a return that is large
    relative to recent volatility.

    return > +k*vol -> +1; return < -k*vol -> -1; otherwise flat.
    """

    def __init__(self, window: int = 20, k: float = 1.5, allow_shorts: bool = True):
        self.window = window
        self.k = k
        self.allow_shorts = allow_shorts

    @property
    def name(self) -> str:
        return f"vol_breakout_{self.window}_{self.k}"

    def generate_signal(self, prices: pd.DataFrame) -> pd.Series:
        ret = SimpleReturns().compute(prices)
        vol = RollingVolatility(self.window, annualise=False).compute(prices)
        band = self.k * vol

        short = -1 if self.allow_shorts else 0
        signal = pd.Series(0.0, index=prices.index)
        signal[ret > band] = 1
        signal[ret < -band] = short

        signal[ret.isna() | vol.isna()] = np.nan
        return self._validate(signal)
