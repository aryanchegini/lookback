from abc import ABC, abstractmethod

import pandas as pd


class Feature(ABC):
    """A feature computes a derived signal from price data.

    Computation MUST be backwards-looking only. compute(prices) MUST NOT
    use any data later than the last row of `prices`.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def compute(self, prices: pd.DataFrame) -> pd.Series: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"
