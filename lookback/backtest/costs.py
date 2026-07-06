from abc import ABC, abstractmethod

import pandas as pd

TRADING_DAYS = 252


class CostModel(ABC):
    """Turns a position series into a per-bar cost series (>= 0).

    Swappable, like Strategy and Sizer: different frictions, one interface.
    """

    @abstractmethod
    def compute(self, position: pd.Series) -> pd.Series: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class NoCost(CostModel):
    """Frictionless — zero cost everywhere."""

    def compute(self, position: pd.Series) -> pd.Series:
        return pd.Series(0.0, index=position.index)


class PerTradeCost(CostModel):
    """Charge on turnover: cost = |Δposition| * cost_bps."""

    def __init__(self, cost_bps: float):
        self.cost_bps = cost_bps

    def compute(self, position: pd.Series) -> pd.Series:
        turnover = position.fillna(0).diff().abs().fillna(0)
        return turnover * (self.cost_bps / 10_000)


class BorrowCost(CostModel):
    """Holding fee on shorts, charged every bar a short is open."""

    def __init__(self, borrow_bps: float):
        self.borrow_bps = borrow_bps

    def compute(self, position: pd.Series) -> pd.Series:
        short_exposure = position.clip(upper=0).abs().fillna(0)
        return short_exposure * (self.borrow_bps / 10_000 / TRADING_DAYS)


class CompositeCost(CostModel):
    """Sum of several cost models (e.g. per-trade + borrow)."""

    def __init__(self, models: list[CostModel]):
        self.models = models

    def compute(self, position: pd.Series) -> pd.Series:
        total = pd.Series(0.0, index=position.index)
        for model in self.models:
            total = total + model.compute(position)
        return total
