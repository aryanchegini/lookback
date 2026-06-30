import numpy as np
import pandas as pd

from lookback.features.moving_average import MovingAverage
from lookback.strategies.base import Strategy


class MovingAverageCrossover(Strategy):
    """Long when the fast MA is above the slow MA; short (or flat) when below."""

    def __init__(self, fast: int = 20, slow: int = 50, allow_shorts: bool = True):
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        self.fast = fast
        self.slow = slow
        self.allow_shorts = allow_shorts

    @property
    def name(self) -> str:
        return f"ma_cross_{self.fast}_{self.slow}"

    def generate_signal(self, prices: pd.DataFrame) -> pd.Series:
        fast = MovingAverage(self.fast).compute(prices)
        slow = MovingAverage(self.slow).compute(prices)

        down = -1 if self.allow_shorts else 0
        signal = pd.Series(
            np.where(fast > slow, 1, down),
            index=prices.index,
            dtype=float,
        )
        # Warm-up: no signal until both MAs exist.
        signal[fast.isna() | slow.isna()] = np.nan
        return self._validate(signal)
