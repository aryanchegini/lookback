import numpy as np
import pandas as pd
import pytest

from lookback.strategies.crossover import MovingAverageCrossover
from lookback.strategies.mean_reversion import MeanReversion
from lookback.strategies.vol_breakout import VolBreakout

ALL = [MovingAverageCrossover(2, 3), MeanReversion(10), VolBreakout(10)]


def _synthetic(n=80):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"close": close}, index=idx)


@pytest.mark.parametrize("strat", ALL)
def test_signals_are_legal(strat):
    sig = strat.generate_signal(_synthetic())
    assert set(sig.dropna().unique()) <= {-1.0, 0.0, 1.0}
    assert sig.index.equals(_synthetic().index)


@pytest.mark.parametrize(
    "strat",
    [
        MovingAverageCrossover(2, 3, allow_shorts=False),
        MeanReversion(10, allow_shorts=False),
        VolBreakout(10, allow_shorts=False),
    ],
)
def test_no_shorts_when_disabled(strat):
    sig = strat.generate_signal(_synthetic())
    assert (sig.dropna() >= 0).all()


@pytest.mark.parametrize("strat", ALL)
def test_strategies_are_backwards_looking(strat):
    p = _synthetic()
    extra = pd.DataFrame({"close": [999.0]}, index=[p.index[-1] + pd.Timedelta(days=1)])
    p2 = pd.concat([p, extra])
    a = strat.generate_signal(p)
    b = strat.generate_signal(p2).iloc[: len(p)]
    pd.testing.assert_series_equal(a, b, check_freq=False)


def test_mean_reversion_fades_extremes():
    # Flat then a sharp spike up -> z-score high -> short.
    close = [100.0] * 15 + [130.0]
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D")
    p = pd.DataFrame({"close": close}, index=idx)
    sig = MeanReversion(window=10, k=1.0).generate_signal(p)
    assert sig.iloc[-1] == -1.0


def test_vol_breakout_follows_big_move():
    # Calm then a large jump -> return >> k*vol -> long.
    close = [100.0 + 0.1 * i for i in range(15)] + [120.0]
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D")
    p = pd.DataFrame({"close": close}, index=idx)
    sig = VolBreakout(window=10, k=1.5).generate_signal(p)
    assert sig.iloc[-1] == 1.0
