from datetime import datetime

import pytest

from lookback.core.orders import Order, OrderSide, OrderType, OrderState
from lookback.core.exceptions import InvalidStateTransitionError


def _order():
    return Order("id1", "AAPL", OrderSide.BUY, 10.0, OrderType.MARKET, datetime(2024, 1, 2))


def test_order_defaults_to_pending():
    order = _order()
    assert order._state is OrderState.PENDING


def test_order_state_is_mutable():
    # not frozen — reassigning _state should work
    order = _order()
    order._state = OrderState.SUBMITTED
    assert order._state == OrderState.SUBMITTED
    
def test_legal_state_transition():
    order = _order()
    order.transition_to(OrderState.SUBMITTED)
    assert order._state == OrderState.SUBMITTED
    
def test_illegal_state_transitions():
    order = _order()
    with pytest.raises(InvalidStateTransitionError):
        order.transition_to(OrderState.FILLED)
    
