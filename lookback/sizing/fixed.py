import pandas as pd

from lookback.sizing.base import Sizer
from lookback.sizing.descriptors import Fraction01


class FixedFraction(Sizer, key="fixed"):
    """Deploy a constant fraction of capital: position = signal * fraction."""

    fraction = Fraction01()

    def __init__(self, fraction: float = 1.0):
        self.fraction = fraction  # validated by the descriptor

    def size(self, signal: pd.Series, prices: pd.DataFrame) -> pd.Series:
        return signal * self.fraction
