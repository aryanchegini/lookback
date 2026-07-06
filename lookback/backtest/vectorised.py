import pandas as pd

from lookback.backtest.costs import (
    BorrowCost,
    CompositeCost,
    CostModel,
    PerTradeCost,
)
from lookback.backtest.result import BacktestResult
from lookback.sizing.base import Sizer
from lookback.strategies.base import Strategy


class VectorisedBacktester:
    """Runs a strategy over a price frame with whole-Series arithmetic.

    The one and only place a signal is lagged into a position lives here
    (position = signal.shift(1)), so no strategy can leak look-ahead bias.
    """

    def __init__(self, cost_bps: float = 0.0, sizer: Sizer | None = None,
                 borrow_bps: float = 0.0, cost_model: CostModel | None = None):
        self.sizer = sizer
        # A CostModel can be supplied directly; otherwise build one from the
        # bps convenience args (per-trade + borrow), keeping behaviour identical.
        if cost_model is not None:
            self.cost_model = cost_model
        else:
            self.cost_model = CompositeCost(
                [PerTradeCost(cost_bps), BorrowCost(borrow_bps)]
            )

    def run(self, strategy: Strategy, prices: pd.DataFrame) -> BacktestResult:
        signal = strategy.generate_signal(prices)

        # Sizing scales the raw signal; still backwards-looking, so the
        # single shift below is all that's needed to avoid look-ahead.
        sized = self.sizer.size(signal, prices) if self.sizer is not None else signal

        # The lag: a decision made from bar t can only be acted on at t+1.
        position = sized.shift(1)

        asset_returns = prices["close"].pct_change()
        gross_returns = position * asset_returns

        costs = self.cost_model.compute(position)

        strategy_returns = gross_returns - costs
        equity_curve = (1 + strategy_returns.fillna(0)).cumprod()

        return BacktestResult(
            strategy_name=strategy.name,
            signal=signal,
            position=position,
            asset_returns=asset_returns,
            costs=costs,
            strategy_returns=strategy_returns,
            equity_curve=equity_curve,
        )
