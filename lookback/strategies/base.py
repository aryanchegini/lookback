from abc import ABC, abstractmethod

import pandas as pd

from lookback.exceptions import InvalidSignalError

_VALID = {-1, 0, 1}


class Strategy(ABC):
    """Turns price data into a signal: an opinion per bar in {-1, 0, +1}.

    -1 short, 0 flat, +1 long. generate_signal(prices) MUST be
    backwards-looking only. It may use the *current* bar; the backtester
    lags the signal by one bar, so the strategy never has to shift itself.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate_signal(self, prices: pd.DataFrame) -> pd.Series: ...

    def _validate(self, signal: pd.Series) -> pd.Series:
        """Reject any value outside {-1, 0, +1} (NaN allowed during warm-up)."""
        bad = set(signal.dropna().unique()) - _VALID
        if bad:
            raise InvalidSignalError(f"{self.name} produced illegal signals: {sorted(bad)}")
        return signal

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"
