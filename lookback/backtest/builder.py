import pandas as pd

from lookback.backtest.result import BacktestResult
from lookback.backtest.vectorised import VectorisedBacktester
from lookback.exceptions import ConfigurationError
from lookback.sizing.base import Sizer, make_sizer
from lookback.strategies.base import Strategy


class BacktestBuilder:
    """Fluent assembly of a backtest run.

    Each with_*/on method sets one piece and returns self, so calls chain:
        BacktestBuilder().with_strategy(s).with_costs(5).on(prices).run()
    """

    def __init__(self):
        self._strategy: Strategy | None = None
        self._prices: pd.DataFrame | None = None
        self._cost_bps: float = 0.0
        self._borrow_bps: float = 0.0
        self._sizer: Sizer | None = None

    def with_strategy(self, strategy: Strategy) -> "BacktestBuilder":
        self._strategy = strategy
        return self

    def with_costs(self, cost_bps: float) -> "BacktestBuilder":
        self._cost_bps = cost_bps
        return self

    def with_borrow(self, borrow_bps: float) -> "BacktestBuilder":
        self._borrow_bps = borrow_bps
        return self

    def with_sizer(self, key: str, **params) -> "BacktestBuilder":
        self._sizer = make_sizer(key, **params)
        return self

    def on(self, prices: pd.DataFrame) -> "BacktestBuilder":
        self._prices = prices
        return self

    def run(self) -> BacktestResult:
        if self._strategy is None:
            raise ConfigurationError("no strategy set (use .with_strategy)")
        if self._prices is None:
            raise ConfigurationError("no prices set (use .on)")
        backtester = VectorisedBacktester(
            cost_bps=self._cost_bps, sizer=self._sizer, borrow_bps=self._borrow_bps
        )
        return backtester.run(self._strategy, self._prices)
