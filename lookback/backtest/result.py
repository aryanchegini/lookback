from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """Immutable record of one backtest run: the pipeline columns + identity."""

    strategy_name: str
    signal: pd.Series
    position: pd.Series
    asset_returns: pd.Series
    strategy_returns: pd.Series
    equity_curve: pd.Series

    @property
    def total_return(self) -> float:
        return float(self.equity_curve.iloc[-1] - 1.0)

    @property
    def n_bars(self) -> int:
        return len(self.equity_curve)

    def __repr__(self) -> str:
        return (
            f"BacktestResult({self.strategy_name}, "
            f"bars={self.n_bars}, total_return={self.total_return:.2%})"
        )
