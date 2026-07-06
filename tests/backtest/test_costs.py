import numpy as np
import pandas as pd
import pytest

from lookback.backtest.costs import (
    BorrowCost,
    CompositeCost,
    NoCost,
    PerTradeCost,
)


def _pos(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def test_no_cost_is_zero():
    out = NoCost().compute(_pos([1, -1, 0, 1]))
    assert (out == 0).all()


def test_per_trade_charges_turnover():
    # flat -> long -> short: turnover 1 then 2; at 10 bps = 0.001 per unit.
    pos = _pos([0, 1, -1])
    out = PerTradeCost(10).compute(pos)
    assert out.iloc[0] == pytest.approx(0.0)
    assert out.iloc[1] == pytest.approx(0.001)   # |1-0| * 10bps
    assert out.iloc[2] == pytest.approx(0.002)   # |-1-1| * 10bps


def test_borrow_only_charges_shorts():
    pos = _pos([1, -1, 0, -1])
    out = BorrowCost(2520).compute(pos)  # 2520/252 = 10 bps/day
    assert out.iloc[0] == 0.0            # long: no borrow
    assert out.iloc[1] == pytest.approx(0.001)
    assert out.iloc[2] == 0.0            # flat
    assert out.iloc[3] == pytest.approx(0.001)


def test_composite_sums_models():
    pos = _pos([0, -1, -1])
    combined = CompositeCost([PerTradeCost(10), BorrowCost(2520)])
    expected = PerTradeCost(10).compute(pos) + BorrowCost(2520).compute(pos)
    pd.testing.assert_series_equal(combined.compute(pos), expected)


def test_backtester_default_matches_bps_args():
    # The refactor must not change numbers: default model == per-trade+borrow.
    from lookback.backtest.vectorised import VectorisedBacktester
    from lookback.strategies.crossover import MovingAverageCrossover

    idx = pd.date_range("2022-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame({"close": 100 + np.cumsum(rng.normal(0, 1, 200))}, index=idx)
    strat = MovingAverageCrossover(10, 30, allow_shorts=True)

    via_args = VectorisedBacktester(cost_bps=5, borrow_bps=100).run(strat, prices)
    via_model = VectorisedBacktester(
        cost_model=CompositeCost([PerTradeCost(5), BorrowCost(100)])
    ).run(strat, prices)
    pd.testing.assert_series_equal(via_args.costs, via_model.costs)
