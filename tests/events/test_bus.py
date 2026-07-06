import gc

import numpy as np
import pandas as pd

from lookback.events.bus import EventBus
from lookback.strategies.crossover import MovingAverageCrossover
from lookback.sweep.grid import param_grid
from lookback.sweep.vanilla import run_sweep


def test_publish_delivers_to_subscriber():
    received = []
    bus = EventBus()

    def handler(payload):
        received.append(payload)

    bus.subscribe("result", handler)
    bus.publish("result", {"x": 1})
    assert received == [{"x": 1}]


def test_weak_ref_drops_dead_subscriber():
    bus = EventBus()

    class Listener:
        def __init__(self):
            self.hits = 0

        def on_result(self, payload):
            self.hits += 1

    listener = Listener()
    bus.subscribe("result", listener.on_result)
    assert bus.subscriber_count("result") == 1

    del listener  # drop the only strong reference
    gc.collect()

    # Bus held only a weak ref -> the listener is gone, not kept alive.
    delivered = bus.publish("result", {"x": 1})
    assert delivered == 0
    assert bus.subscriber_count("result") == 0


def test_sweep_publishes_results():
    idx = pd.date_range("2022-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame({"close": 100 + np.cumsum(rng.normal(0, 1, 200))}, index=idx)

    seen = []
    bus = EventBus()

    def collect(row):
        seen.append(row)

    bus.subscribe("result", collect)

    def build(fast, slow):
        return MovingAverageCrossover(fast, slow)

    run_sweep(build, prices, param_grid(fast=[5, 10], slow=[50]), bus=bus)
    assert len(seen) == 2
