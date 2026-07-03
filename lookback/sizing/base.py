from abc import ABC, abstractmethod

import pandas as pd

from lookback.core.exceptions import ConfigurationError


class Sizer(ABC):
    """Scales a raw {-1,0,+1} signal into a position size.

    size(signal, prices) MUST stay backwards-looking; the backtester lags
    the sized signal by one bar, same as an unsized one.
    """

    registry: dict[str, type["Sizer"]] = {}

    def __init_subclass__(cls, key: str | None = None, **kwargs):
        # Runs automatically when a subclass is defined. Auto-registers it
        # under `key` so a factory can build sizers by name.
        super().__init_subclass__(**kwargs)
        if key is not None:
            Sizer.registry[key] = cls

    @abstractmethod
    def size(self, signal: pd.Series, prices: pd.DataFrame) -> pd.Series: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def make_sizer(key: str, **params) -> Sizer:
    """Build a registered sizer by name."""
    if key not in Sizer.registry:
        raise ConfigurationError(
            f"unknown sizer {key!r}; known: {sorted(Sizer.registry)}"
        )
    return Sizer.registry[key](**params)
