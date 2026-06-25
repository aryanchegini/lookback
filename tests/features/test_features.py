import numpy as np
import pandas as pd
import pytest

from lookback.features.returns import SimpleReturns, LogReturns
from lookback.features.volatility import RollingVolatility
from lookback.features.moving_average import MovingAverage
from lookback.features.zscore import ZScore
from lookback.features.momentum import Momentum


def _synthetic(n=80):
    """Deterministic random-walk close series for the rolling features."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"close": close}, index=idx)


# ---------- returns ----------

def test_simple_returns():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    p = pd.DataFrame({"close": [100.0, 110.0, 99.0]}, index=idx)
    out = SimpleReturns().compute(p)
    assert SimpleReturns().name == "simple_returns"
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(0.10)
    assert out.iloc[2] == pytest.approx(99 / 110 - 1)
    assert out.index.equals(p.index)


def test_log_returns():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    p = pd.DataFrame({"close": [100.0, 110.0, 99.0]}, index=idx)
    out = LogReturns().compute(p)
    assert LogReturns().name == "log_returns"
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(np.log(110 / 100))
    assert out.iloc[2] == pytest.approx(np.log(99 / 110))


# ---------- volatility ----------

def test_rolling_volatility():
    p = _synthetic()
    raw = RollingVolatility(window=20, annualise=False).compute(p)
    ann = RollingVolatility(window=20, annualise=True).compute(p)
    assert RollingVolatility(window=20).name == "rolling_vol_20"
    # 1 row lost to the diff + 19 to fill the window => first 20 are NaN
    assert raw.iloc[:20].isna().all()
    assert not np.isnan(raw.iloc[-1])
    assert np.allclose((ann / raw).dropna(), np.sqrt(252))
    assert raw.index.equals(p.index)


# ---------- moving average ----------

def test_moving_average():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    p = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    out = MovingAverage(window=3).compute(p)
    assert MovingAverage(window=3).name == "sma_3"
    assert out.iloc[:2].isna().all()  # window-1 leading NaNs
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[3] == pytest.approx(3.0)
    assert out.iloc[4] == pytest.approx(4.0)
    assert out.index.equals(p.index)


# ---------- zscore ----------

def test_zscore():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    p = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    out = ZScore(window=3).compute(p)
    assert ZScore(window=3).name == "zscore_3"
    # window [1,2,3]: mean 2, sample std (ddof=1) = 1 -> z = (3-2)/1 = 1
    assert out.iloc[-1] == pytest.approx(1.0)
    assert out.iloc[:2].isna().all()
    assert out.index.equals(p.index)


# ---------- momentum ----------

def test_momentum():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    p = pd.DataFrame({"close": [100.0, 0, 0, 0, 110.0]}, index=idx)
    out = Momentum(window=4).compute(p)
    assert Momentum(window=20).name == "momentum_20"
    assert out.iloc[-1] == pytest.approx(0.10)  # 110 vs 100, 4 bars back
    assert out.iloc[:4].isna().all()  # first `window` NaN
    assert out.index.equals(p.index)


# ---------- shared contract ----------

def test_repr_uses_name():
    assert repr(MovingAverage(50)) == "MovingAverage(sma_50)"


def test_features_are_backwards_looking():
    """Appending a future bar must not change already-computed values."""
    p = _synthetic()
    extra = pd.DataFrame(
        {"close": [999.0]}, index=[p.index[-1] + pd.Timedelta(days=1)]
    )
    p2 = pd.concat([p, extra])
    for feat in (SimpleReturns(), LogReturns(), RollingVolatility(20),
                 MovingAverage(20), ZScore(20), Momentum(20)):
        a = feat.compute(p)
        b = feat.compute(p2).iloc[: len(p)]
        pd.testing.assert_series_equal(a, b, check_freq=False)
