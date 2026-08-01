"""A Facade: one small surface over the data, backtest, and sweep subsystems,
so common tasks don't require wiring the pieces together by hand.
"""

from datetime import datetime

import pandas as pd

from lookback.backtest.portfolio import PortfolioBacktester, PortfolioResult
from lookback.backtest.result import BacktestResult
from lookback.backtest.vectorised import VectorisedBacktester
from lookback.exceptions import ConfigurationError
from lookback.data.cache import ParquetCache
from lookback.data.source import DataSource
from lookback.data.store import DataStore
from lookback.sizing.base import Sizer
from lookback.strategies.base import Strategy
from lookback.sweep.grid import param_grid
from lookback.sweep.vanilla import run_sweep


class Lookback:
    """Convenience front-end. Pass a DataSource to enable price loading;
    backtesting and sweeps work on any price frame without one.
    """

    def __init__(self, data_source: DataSource | None = None, cache: ParquetCache | None = None):
        self._store = DataStore(data_source, cache) if data_source is not None else None

    def prices(self, symbol: str, as_of: datetime, start: datetime | None = None):
        
        if self._store is None:
            raise ConfigurationError("no data source configured")
        return self._store.get_bars(symbol, as_of, start)

    def backtest(self, strategy: Strategy, prices: pd.DataFrame, *, cost_bps: float = 0.0, borrow_bps: float = 0.0, sizer: Sizer | None = None) -> BacktestResult:
        return VectorisedBacktester(cost_bps=cost_bps, borrow_bps=borrow_bps, sizer=sizer).run(strategy, prices)

    def portfolio(self, strategy: Strategy, prices_by_symbol: dict[str, pd.DataFrame], *, cost_bps: float = 0.0) -> PortfolioResult:
        
        engine = PortfolioBacktester(VectorisedBacktester(cost_bps=cost_bps))
        
        return engine.run(strategy, prices_by_symbol)

    def sweep(self, build, prices: pd.DataFrame, *, cost_bps: float = 0.0, rank_by: str = "sharpe", **axes) -> pd.DataFrame:
        return run_sweep(build, prices, param_grid(**axes), cost_bps=cost_bps, rank_by=rank_by)
