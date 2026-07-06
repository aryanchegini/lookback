from datetime import datetime

import numpy as np
import pandas as pd

from lookback.backtest.vectorised import VectorisedBacktester
from lookback.data.source import DataSource
from lookback.facade import Lookback
from lookback.strategies.crossover import MovingAverageCrossover
from lookback.sweep.grid import param_grid
from lookback.sweep.vanilla import run_sweep


def _prices(n=200):
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"close": 100 + np.cumsum(rng.normal(0, 1, n))}, index=idx)


def build_cross(fast, slow):
    return MovingAverageCrossover(fast, slow)


def test_facade_backtest_matches_direct():
    prices = _prices()
    strat = MovingAverageCrossover(10, 30)
    via_facade = Lookback().backtest(strat, prices, cost_bps=5)
    direct = VectorisedBacktester(cost_bps=5).run(strat, prices)
    pd.testing.assert_series_equal(via_facade.equity_curve, direct.equity_curve)


def test_facade_sweep_matches_direct():
    prices = _prices()
    via_facade = Lookback().sweep(build_cross, prices, fast=[5, 10], slow=[50, 100])
    direct = run_sweep(build_cross, prices, param_grid(fast=[5, 10], slow=[50, 100]))
    pd.testing.assert_frame_equal(via_facade, direct)


def test_facade_prices_via_source():
    class _Source(DataSource):
        def fetch_bars(self, symbol, start, end):
            idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
            return pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)

    lb = Lookback(_Source())
    bars = lb.prices("X", as_of=datetime(2024, 1, 2))
    assert len(bars) == 2  # as-of slice drops Jan 3
