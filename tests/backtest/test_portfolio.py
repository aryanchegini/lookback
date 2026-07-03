import numpy as np
import pandas as pd
import pytest

from lookback.backtest.portfolio import PortfolioBacktester
from lookback.core.exceptions import InsufficientDataError
from lookback.strategies.crossover import MovingAverageCrossover


def _prices(seed, n=250):
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"close": close}, index=idx)


def test_empty_raises():
    with pytest.raises(InsufficientDataError):
        PortfolioBacktester().run(MovingAverageCrossover(10, 30), {})


def test_portfolio_is_equal_weight_mean():
    data = {"AAA": _prices(1), "BBB": _prices(2)}
    strat = MovingAverageCrossover(10, 30)
    result = PortfolioBacktester().run(strat, data)

    a = result.per_symbol["AAA"].strategy_returns
    b = result.per_symbol["BBB"].strategy_returns
    expected = pd.concat([a, b], axis=1).mean(axis=1)
    pd.testing.assert_series_equal(result.portfolio_returns, expected, check_names=False)


def test_portfolio_equity_from_returns():
    data = {"AAA": _prices(1), "BBB": _prices(2), "CCC": _prices(3)}
    result = PortfolioBacktester().run(MovingAverageCrossover(10, 30), data)
    expected = (1 + result.portfolio_returns.fillna(0)).cumprod()
    pd.testing.assert_series_equal(result.equity_curve, expected)
    assert set(result.per_symbol) == {"AAA", "BBB", "CCC"}


def test_diversification_lowers_volatility():
    # Portfolio vol should not exceed the worst single-name vol.
    data = {f"S{i}": _prices(i) for i in range(5)}
    strat = MovingAverageCrossover(10, 30)
    result = PortfolioBacktester().run(strat, data)

    port_vol = result.portfolio_returns.std()
    single_vols = [r.strategy_returns.std() for r in result.per_symbol.values()]
    assert port_vol <= max(single_vols)
