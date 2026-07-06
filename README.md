# lookback

A point-in-time market-data store and vectorised backtester for trading strategies, written in Python.

`lookback` pulls historical market data, builds signals from it, runs a strategy over that data, and tells you how it would have done — with transaction costs, short-borrow costs, and position sizing factored in. You can sweep over a range of parameters (in parallel, with resumable checkpoints) to check whether a strategy is genuinely good or just got lucky on one setting.

The thing I most wanted to get right is that a strategy should never be able to see the future. Every data lookup is tied to an `as_of` date and only returns data up to that point, so there's no way to accidentally trade on information you wouldn't have had at the time. That kind of leak (look-ahead bias) is the usual reason a backtest looks great and then falls apart live, so it's handled structurally rather than by being careful: the whole pipeline is backwards-looking, and the one place a signal becomes a tradable position — a single `shift(1)` — lives in the backtester and nowhere else.

## How a backtest flows

```
DataStore.get_bars(as_of)      point-in-time prices (never past as_of)
   → Feature.compute           backwards-looking inputs (rolling stats)
   → Strategy.generate_signal   an opinion per bar in {-1, 0, +1}
   → Sizer.size                scale the opinion into a position size
   → position = shift(1)       the one look-ahead guard: act at t+1
   → × asset returns − costs   per-bar P&L, net of trade + borrow costs
   → equity curve → metrics    Sharpe, drawdown, CAGR, win rate
```

`BacktestBuilder` assembles a run fluently; `PortfolioBacktester` runs one strategy across many symbols and equal-weights them; the sweep repeats a run over a parameter grid.

## Getting started

Requires Python 3.13+.

```bash
git clone <your-fork-url> lookback
cd lookback
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the tests:

```bash
pytest          # or: make test
```

A one-call entry point over the subsystems:

```python
from lookback.facade import Lookback
from lookback.strategies.crossover import MovingAverageCrossover

lb = Lookback()
result = lb.backtest(MovingAverageCrossover(20, 50), prices, cost_bps=5)
print(result.summary())        # Sharpe, drawdown, CAGR, win rate
```

## Layout

```
lookback/
├── core/         # exceptions, instruments (Bar), orders (state machine)
├── data/         # data sources, point-in-time store, partitioned parquet cache
├── features/     # derived signals (returns, vol, moving averages, zscore, momentum)
├── strategies/   # trading strategies (crossover, mean-reversion, vol-breakout)
├── sizing/       # position sizers + validating descriptors + factory
├── backtest/     # vectorised backtester, cost models, metrics, builder, portfolio
├── sweep/        # lazy param grid + vanilla and parallel/resumable runners
├── events/       # weakref EventBus (Observer)
├── utils/        # decorators (timing, memoisation)
└── facade.py     # one-call front end over the subsystems
```

## Tech stack

Python 3.13 · pandas · NumPy · PyArrow (parquet) · pytest
