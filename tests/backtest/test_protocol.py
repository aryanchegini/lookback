from lookback.backtest.protocol import Backtester
from lookback.backtest.vectorised import VectorisedBacktester


def test_vectorised_satisfies_protocol_structurally():
    # Conforms via its run() method, without inheriting Backtester.
    assert isinstance(VectorisedBacktester(), Backtester)
    assert Backtester not in VectorisedBacktester.__mro__


def test_non_conforming_object_is_rejected():
    class NotABacktester:
        def something_else(self):
            return None

    assert not isinstance(NotABacktester(), Backtester)
