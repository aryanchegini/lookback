from dataclasses import dataclass

import pandas as pd

from lookback.backtest.metrics import cagr, max_drawdown, sharpe_ratio, win_rate


@dataclass(frozen=True)
class BacktestResult:
    """Immutable record of one backtest run: the pipeline columns + identity."""

    strategy_name: str
    signal: pd.Series
    position: pd.Series
    asset_returns: pd.Series
    costs: pd.Series
    strategy_returns: pd.Series  # net of costs
    equity_curve: pd.Series

    @property
    def total_return(self) -> float:
        return float(self.equity_curve.iloc[-1] - 1.0)

    @property
    def n_bars(self) -> int:
        return len(self.equity_curve)

    def summary(self) -> dict[str, float]:
        """Risk-adjusted scorecard for this run."""
        return {"total_return": self.total_return, "cagr": cagr(self.equity_curve), "sharpe": sharpe_ratio(self.strategy_returns), "max_drawdown": max_drawdown(self.equity_curve), "win_rate": win_rate(self.strategy_returns)}

    def __repr__(self) -> str:
        return (f"BacktestResult({self.strategy_name}, " f"bars={self.n_bars}, total_return={self.total_return:.2%})")
