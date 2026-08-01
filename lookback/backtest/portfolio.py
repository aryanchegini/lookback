from dataclasses import dataclass

import pandas as pd

from lookback.backtest.metrics import cagr, max_drawdown, sharpe_ratio, win_rate
from lookback.backtest.result import BacktestResult
from lookback.backtest.vectorised import VectorisedBacktester
from lookback.exceptions import InsufficientDataError
from lookback.strategies.base import Strategy


@dataclass(frozen=True)
class PortfolioResult:
    """Combined result of running one strategy across several instruments."""

    strategy_name: str
    per_symbol: dict[str, BacktestResult]
    portfolio_returns: pd.Series
    equity_curve: pd.Series

    @property
    def total_return(self) -> float:
        return float(self.equity_curve.iloc[-1] - 1.0)

    def summary(self) -> dict[str, float]:
        return {"total_return": self.total_return, "cagr": cagr(self.equity_curve), "sharpe": sharpe_ratio(self.portfolio_returns), "max_drawdown": max_drawdown(self.equity_curve), "win_rate": win_rate(self.portfolio_returns)}

    def __repr__(self) -> str:
        return (f"PortfolioResult({self.strategy_name}, " f"symbols={len(self.per_symbol)}, total_return={self.total_return:.2%})")


class PortfolioBacktester:
    """Runs a strategy across many symbols and equal-weights the returns."""

    def __init__(self, backtester: VectorisedBacktester | None = None):
        self.backtester = backtester or VectorisedBacktester()

    def run(self, strategy: Strategy, prices_by_symbol: dict[str, pd.DataFrame]) -> PortfolioResult:
        if not prices_by_symbol:
            raise InsufficientDataError("no symbols provided")

        per_symbol = {
            sym: self.backtester.run(strategy, prices) for sym, prices in prices_by_symbol.items()
        }

        # One column of strategy returns per symbol, aligned on the union index.
        returns = pd.DataFrame(
            {sym: r.strategy_returns for sym, r in per_symbol.items()}
        )
        # Equal weight: mean across symbols (skipping symbols still warming up).
        portfolio_returns = returns.mean(axis=1)
        equity_curve = (1 + portfolio_returns.fillna(0)).cumprod()

        return PortfolioResult(strategy_name=strategy.name, per_symbol=per_symbol, portfolio_returns=portfolio_returns, equity_curve=equity_curve)
