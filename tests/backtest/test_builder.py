import numpy as np
import pandas as pd
import pytest

from lookback.backtest.builder import BacktestBuilder
from lookback.backtest.vectorised import VectorisedBacktester
from lookback.core.exceptions import ConfigurationError
from lookback.sizing import VolTarget
from lookback.strategies.crossover import MovingAverageCrossover


def _synthetic(n=120):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"close": close}, index=idx)


def test_chaining_returns_self():
    b = BacktestBuilder()
    assert b.with_costs(5) is b
    assert b.on(_synthetic()) is b


def test_build_matches_direct_construction():
    prices = _synthetic()
    strat = MovingAverageCrossover(10, 30)

    built = (
        BacktestBuilder()
        .with_strategy(strat)
        .with_costs(5)
        .with_sizer("vol_target", target_vol=0.15, window=20)
        .on(prices)
        .run()
    )
    direct = VectorisedBacktester(
        cost_bps=5, sizer=VolTarget(target_vol=0.15, window=20)
    ).run(strat, prices)

    pd.testing.assert_series_equal(built.equity_curve, direct.equity_curve)


def test_missing_strategy_raises():
    with pytest.raises(ConfigurationError):
        BacktestBuilder().on(_synthetic()).run()


def test_missing_prices_raises():
    with pytest.raises(ConfigurationError):
        BacktestBuilder().with_strategy(MovingAverageCrossover(10, 30)).run()
