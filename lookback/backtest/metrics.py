import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS, risk_free: float = 0.0) -> float:
    """Annualised return per unit of risk. risk_free is per-period."""
    excess = returns.dropna() - risk_free
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / sd)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Worst peak-to-trough decline of the equity curve (a negative number)."""
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def cagr(equity_curve: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate implied by the equity curve."""
    n = len(equity_curve)
    if n < 2:
        return 0.0
    total_growth = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = n / periods_per_year
    return float(total_growth ** (1 / years) - 1.0)


def win_rate(returns: pd.Series) -> float:
    """Fraction of bars with a positive return."""
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())
