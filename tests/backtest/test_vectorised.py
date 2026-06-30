import numpy as np
import pandas as pd

from lookback.backtest.vectorised import VectorisedBacktester
from lookback.strategies.base import Strategy


def _synthetic(n=60):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"close": close}, index=idx)


class _PerfectForesight(Strategy):
    """CHEAT: signal = sign of the *current* bar's return.

    Without the backtester's shift(1), position*return would be
    sign(r)*r = |r| >= 0 on every bar -> impossibly perfect.
    """

    @property
    def name(self) -> str:
        return "perfect_foresight"

    def generate_signal(self, prices: pd.DataFrame) -> pd.Series:
        return self._validate(np.sign(prices["close"].pct_change()))


class _AlwaysLong(Strategy):
    @property
    def name(self) -> str:
        return "always_long"

    def generate_signal(self, prices: pd.DataFrame) -> pd.Series:
        return self._validate(pd.Series(1.0, index=prices.index))


def test_no_lookahead_via_perfect_foresight():
    prices = _synthetic()
    result = VectorisedBacktester().run(_PerfectForesight(), prices)
    pnl = result.strategy_returns.dropna()
    # If shift(1) were removed, every bar would be >= 0. It must not be.
    assert (pnl < 0).any(), "backtester leaked look-ahead: cheat never loses"


def test_position_is_lagged():
    prices = _synthetic()
    result = VectorisedBacktester().run(_AlwaysLong(), prices)
    expected = result.signal.shift(1)
    pd.testing.assert_series_equal(result.position, expected, check_names=False)


def test_equity_matches_returns():
    prices = _synthetic()
    result = VectorisedBacktester().run(_AlwaysLong(), prices)
    asset = prices["close"].pct_change()
    # Long every bar (after the 1-bar warm-up) => strategy return == asset return.
    pd.testing.assert_series_equal(
        result.strategy_returns.iloc[1:], asset.iloc[1:], check_names=False
    )
    expected_equity = (1 + result.strategy_returns.fillna(0)).cumprod()
    pd.testing.assert_series_equal(result.equity_curve, expected_equity, check_names=False)


def test_backtest_is_backwards_looking():
    prices = _synthetic()
    extra = pd.DataFrame(
        {"close": [999.0]}, index=[prices.index[-1] + pd.Timedelta(days=1)]
    )
    prices2 = pd.concat([prices, extra])
    bt = VectorisedBacktester()
    a = bt.run(_PerfectForesight(), prices).strategy_returns
    b = bt.run(_PerfectForesight(), prices2).strategy_returns.iloc[: len(prices)]
    pd.testing.assert_series_equal(a, b, check_freq=False)
