import pandas as pd

from lookback.backtest.result import BacktestResult
from lookback.strategies.base import Strategy


class VectorisedBacktester:
    """Runs a strategy over a price frame with whole-Series arithmetic.

    The one and only place a signal is lagged into a position lives here
    (position = signal.shift(1)), so no strategy can leak look-ahead bias.
    """

    def run(self, strategy: Strategy, prices: pd.DataFrame) -> BacktestResult:
        signal = strategy.generate_signal(prices)

        # The lag: a decision made from bar t can only be acted on at t+1.
        position = signal.shift(1)

        asset_returns = prices["close"].pct_change()
        strategy_returns = position * asset_returns
        equity_curve = (1 + strategy_returns.fillna(0)).cumprod()

        return BacktestResult(
            strategy_name=strategy.name,
            signal=signal,
            position=position,
            asset_returns=asset_returns,
            strategy_returns=strategy_returns,
            equity_curve=equity_curve,
        )
