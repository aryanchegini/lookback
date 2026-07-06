from collections.abc import Callable, Iterable

import pandas as pd

from lookback.backtest.vectorised import VectorisedBacktester
from lookback.events.bus import EventBus
from lookback.strategies.base import Strategy


def run_sweep(build: Callable[..., Strategy], prices: pd.DataFrame,
              grid: Iterable[dict], *, cost_bps: float = 0.0,
              rank_by: str = "sharpe", bus: EventBus | None = None) -> pd.DataFrame:
    """Run one backtest per param combo; return a table ranked by `rank_by`.

    `build(**params)` maps a config dict to a Strategy. Combos that are
    invalid (build raises ValueError) are skipped, not fatal. If a `bus` is
    given, each completed combo is published as a "result" event.
    """
    backtester = VectorisedBacktester(cost_bps=cost_bps)
    rows = []
    for params in grid:
        try:
            strategy = build(**params)
        except ValueError:
            continue  # e.g. fast >= slow; skip this combo
        result = backtester.run(strategy, prices)
        row = {**params, **result.summary()}
        rows.append(row)
        if bus is not None:
            bus.publish("result", row)

    df = pd.DataFrame(rows)
    if rank_by in df.columns:
        df = df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    return df
