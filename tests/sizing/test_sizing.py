import numpy as np
import pandas as pd
import pytest

from lookback.exceptions import ConfigurationError
from lookback.sizing import FixedFraction, Sizer, VolTarget, make_sizer
from lookback.sizing.descriptors import Fraction01, PositiveInt


def _synthetic(n=80):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"close": close}, index=idx)


# ---------- descriptors ----------

@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
def test_fraction01_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        FixedFraction(fraction=bad)


@pytest.mark.parametrize("bad", [0, -3, 2.5, True])
def test_positive_int_rejects_bad(bad):
    with pytest.raises(ValueError):
        VolTarget(window=bad)


def test_descriptor_validates_on_reassignment():
    f = FixedFraction(0.5)
    with pytest.raises(ValueError):
        f.fraction = 5.0  # not just at construction


# ---------- sizing behaviour ----------

def test_fixed_fraction_scales():
    sig = pd.Series([1.0, -1.0, 0.0])
    out = FixedFraction(0.5).size(sig, None)
    pd.testing.assert_series_equal(out, sig * 0.5)


def test_vol_target_caps_leverage():
    p = _synthetic()
    sig = pd.Series(1.0, index=p.index)
    out = VolTarget(target_vol=0.15, window=20, max_leverage=3.0).size(sig, p)
    assert out.dropna().abs().max() <= 3.0 + 1e-9


def test_vol_target_is_backwards_looking():
    p = _synthetic()
    sig = pd.Series(1.0, index=p.index)
    extra = pd.DataFrame({"close": [999.0]}, index=[p.index[-1] + pd.Timedelta(days=1)])
    p2 = pd.concat([p, extra])
    sizer = VolTarget(window=20)
    a = sizer.size(sig, p)
    b = sizer.size(pd.Series(1.0, index=p2.index), p2).iloc[: len(p)]
    pd.testing.assert_series_equal(a, b, check_freq=False)


# ---------- registry / __init_subclass__ ----------

def test_subclasses_auto_registered():
    assert Sizer.registry["fixed"] is FixedFraction
    assert Sizer.registry["vol_target"] is VolTarget


def test_make_sizer_builds_by_name():
    s = make_sizer("fixed", fraction=0.25)
    assert isinstance(s, FixedFraction)
    assert s.fraction == 0.25


def test_make_sizer_unknown_raises():
    with pytest.raises(ConfigurationError):
        make_sizer("nope")
