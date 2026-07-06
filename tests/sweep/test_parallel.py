import numpy as np
import pandas as pd

from lookback.strategies.crossover import MovingAverageCrossover
from lookback.sweep.grid import param_grid
from lookback.sweep.parallel import pending_params, run_sweep_parallel
from lookback.sweep.vanilla import run_sweep


def build_cross(fast, slow):
    """Top-level factory: picklable for the process pool."""
    return MovingAverageCrossover(fast, slow)


def _prices(n=200):
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"close": 100 + np.cumsum(rng.normal(0, 1, n))}, index=idx)


def test_parallel_matches_serial(tmp_path):
    prices = _prices()
    axes = dict(fast=[5, 10, 20], slow=[50, 100])
    serial = run_sweep(build_cross, prices, param_grid(**axes), rank_by="sharpe")
    parallel = run_sweep_parallel(
        build_cross, prices, param_grid(**axes), tmp_path, processes=2
    )
    # Same combos, same metrics (compare sorted by params).
    key = ["fast", "slow"]
    s = serial.sort_values(key).reset_index(drop=True)
    p = parallel.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(s[key + ["sharpe"]], p[key + ["sharpe"]])


def test_resume_skips_completed(tmp_path):
    prices = _prices()
    first = list(param_grid(fast=[5, 10], slow=[50]))
    run_sweep_parallel(build_cross, prices, first, tmp_path, processes=2)

    # A larger grid overlapping the first: only the new combo should be pending.
    bigger = list(param_grid(fast=[5, 10, 20], slow=[50]))
    todo = pending_params(bigger, tmp_path)
    assert todo == [{"fast": 20, "slow": 50}]


def test_second_run_returns_full_table(tmp_path):
    prices = _prices()
    run_sweep_parallel(build_cross, prices, param_grid(fast=[5, 10], slow=[50]),
                       tmp_path, processes=2)
    df = run_sweep_parallel(build_cross, prices, param_grid(fast=[5, 10, 20], slow=[50]),
                            tmp_path, processes=2)
    assert len(df) == 3  # 2 resumed from checkpoint + 1 new
