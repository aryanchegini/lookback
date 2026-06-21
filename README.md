# lookback

A backtester for trading strategies, written in Python.

`lookback` pulls historical market data, builds signals from it, runs a strategy over that data, and tells you how it would have done, with transaction costs and position sizing factored in. You can also sweep over a range of parameters to check whether a strategy is genuinely good or just got lucky on one setting.

The thing I most wanted to get right is that a strategy should never be able to see the future. Every data lookup is tied to an `as_of` date and only returns data up to that point, so there's no way to accidentally trade on information you wouldn't have had at the time. That kind of leak (look-ahead bias) is the usual reason a backtest looks great and then falls apart live, so I handle it in the data layer rather than relying on carefulness.

## Design

Some notes on how it's put together:

- **Point-in-time data store** — every market-data query is bounded by `as_of`, so the look-ahead guarantee comes from the data boundary itself instead of me remembering to be careful.
- **Vectorised backtester** - the simulation is just a handful of aligned pandas operations. The fiddly part is the `shift()` so a signal at time *T* affects returns at *T+1* and not the same bar.
- **Costs and sizing** — transaction costs (bps + fixed) and position sizing (fixed-fraction and vol-targeting) are built in, since a strategy that doesn't survive costs isn't worth much.
- **ABCs vs Protocols** — features share behaviour, so `Feature` is an ABC; backtesters only share an interface, so `Backtester` is a `Protocol`.
- **Domain model** — immutable, slotted `Bar` records so millions fit in memory, an `Order` lifecycle as an explicit state machine, and a small exception hierarchy so I can catch domain errors without swallowing real bugs.

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

## Layout

```
lookback/
├── core/         # exceptions, instruments, orders, events, time helpers
├── data/         # data sources + point-in-time store + caching
├── features/     # derived signals (returns, vol, moving averages, ...)
├── strategies/   # trading strategies
├── backtest/     # vectorised backtester, costs, sizing, metrics
└── sweep/        # parallel parameter sweeps
```

## Tech stack

Python 3.13 · pandas · NumPy · PyArrow (parquet) · pytest
