import pandas as pd

from lookback.features.volatility import RollingVolatility
from lookback.sizing.base import Sizer
from lookback.sizing.descriptors import Fraction01, PositiveInt


class VolTarget(Sizer, key="vol_target"):
    """Scale the bet to hold risk roughly constant.

    position = signal * (target_vol / recent_vol), capped at max_leverage.
    Calm markets -> size up; wild markets -> size down.
    """

    window = PositiveInt()
    target_vol = Fraction01()

    def __init__(self, target_vol: float = 0.15, window: int = 20,
                 max_leverage: float = 3.0):
        self.target_vol = target_vol   # annualised, e.g. 0.15 = 15%
        self.window = window
        self.max_leverage = max_leverage

    def size(self, signal: pd.Series, prices: pd.DataFrame) -> pd.Series:
        recent_vol = RollingVolatility(self.window, annualise=True).compute(prices)
        leverage = (self.target_vol / recent_vol).clip(upper=self.max_leverage)
        return signal * leverage
