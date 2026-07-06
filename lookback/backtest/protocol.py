from typing import Protocol, runtime_checkable

import pandas as pd

from lookback.backtest.result import BacktestResult
from lookback.strategies.base import Strategy


@runtime_checkable
class Backtester(Protocol):
    """Structural contract for anything that can run a strategy.

    Deliberately a Protocol, not an ABC: conformance is by *shape*, not
    inheritance. VectorisedBacktester and PortfolioBacktester satisfy this
    without importing it — the ABC (Strategy/Feature/Sizer/CostModel) vs
    Protocol split the project leans on.
    """

    def run(self, strategy: Strategy, prices: pd.DataFrame) -> BacktestResult: ...
