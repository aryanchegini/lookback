import types

import numpy as np
import pandas as pd

from lookback.strategies.crossover import MovingAverageCrossover
from lookback.sweep.grid import param_grid
from lookback.sweep.vanilla import run_sweep


def build_cross(fast, slow):
    """Top-level factory (picklable for Sunday's process pool)."""
    return MovingAverageCrossover(fast, slow)


def _prices(n=300):
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"close": 100 + np.cumsum(rng.normal(0, 1, n))}, index=idx)


def test_param_grid_is_lazy_generator():
    g = param_grid(fast=[10, 20], slow=[50, 100])
    assert isinstance(g, types.GeneratorType)


def test_param_grid_enumerates_all_combos():
    combos = list(param_grid(fast=[10, 20], slow=[50, 100, 200]))
    assert len(combos) == 6
    assert {"fast": 10, "slow": 50} in combos
    assert {"fast": 20, "slow": 200} in combos


def test_run_sweep_ranks_by_metric():
    grid = param_grid(fast=[5, 10, 20], slow=[30, 60])
    df = run_sweep(build_cross, _prices(), grid, cost_bps=5, rank_by="sharpe")
    assert list(df.columns[:2]) == ["fast", "slow"]
    # Sorted descending by sharpe.
    assert df["sharpe"].is_monotonic_decreasing


def test_run_sweep_skips_invalid_combos():
    grid = param_grid(fast=[10, 20], slow=[10, 50])
    df = run_sweep(build_cross, _prices(), grid)
    # (10,10) and (20,10) are invalid (fast >= slow) -> only 2 valid rows.
    assert len(df) == 2
    assert (df["fast"] < df["slow"]).all()
