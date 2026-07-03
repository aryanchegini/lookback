import numpy as np
import pandas as pd
import pytest

from lookback.backtest.metrics import cagr, max_drawdown, sharpe_ratio, win_rate


def test_sharpe_zero_when_no_variance():
    r = pd.Series([0.01, 0.01, 0.01])
    assert sharpe_ratio(r) == 0.0


def test_sharpe_known_value():
    r = pd.Series([0.01, -0.01, 0.02, 0.0])
    expected = np.sqrt(252) * r.mean() / r.std(ddof=1)
    assert sharpe_ratio(r) == pytest.approx(expected)


def test_max_drawdown():
    equity = pd.Series([1.0, 1.2, 0.9, 1.0, 1.5])
    # Worst drop: peak 1.2 -> trough 0.9 => 0.9/1.2 - 1 = -0.25.
    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_cagr_doubling_in_one_period_block():
    equity = pd.Series([1.0, 2.0])
    # 2 points, 2 periods/year => 1 year, doubled => 100%.
    assert cagr(equity, periods_per_year=2) == pytest.approx(1.0)


def test_win_rate_ignores_nan():
    r = pd.Series([0.01, -0.01, 0.02, np.nan, -0.03])
    # 2 positives out of 4 non-NaN bars.
    assert win_rate(r) == pytest.approx(0.5)
