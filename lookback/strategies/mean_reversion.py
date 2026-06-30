import numpy as np
import pandas as pd

from lookback.features.zscore import ZScore
from lookback.strategies.base import Strategy


class MeanReversion(Strategy):
    """Fade extremes: buy when price is unusually low, sell when unusually high.

    Uses a rolling z-score. z < -k -> +1 (cheap, bet it reverts up);
    z > +k -> -1 (expensive, bet it reverts down); otherwise flat.
    """

    def __init__(self, window: int = 60, k: float = 1.0, allow_shorts: bool = True):
        self.window = window
        self.k = k
        self.allow_shorts = allow_shorts

    @property
    def name(self) -> str:
        return f"mean_rev_{self.window}_{self.k}"

    def generate_signal(self, prices: pd.DataFrame) -> pd.Series:
        z = ZScore(self.window).compute(prices)

        short = -1 if self.allow_shorts else 0
        signal = pd.Series(0.0, index=prices.index)
        signal[z < -self.k] = 1
        signal[z > self.k] = short

        signal[z.isna()] = np.nan
        return self._validate(signal)
