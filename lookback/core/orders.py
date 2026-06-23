from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from lookback.core.exceptions import InvalidStateTransitionError


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"
    
    
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    

class OrderState(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    
    
# pending -> submitted, cancelled, rejected (pre-submit rejection, your own checks refuse sending it out to the exchange)
# submitted -> filled, partially_filled, cancelled, rejected (rejected by exchange)
# partially_filled -> cancelled, filled (partially filled to rejected is very rare, hence not included)
# filled -> none
# cancelled -> none
# rejected -> none
    
_LEGAL_TRANSITIONS: dict[OrderState, set[OrderState]] = {
      OrderState.PENDING:          {OrderState.SUBMITTED, OrderState.CANCELLED, OrderState.REJECTED},
      OrderState.SUBMITTED:        {OrderState.FILLED, OrderState.PARTIALLY_FILLED, OrderState.CANCELLED, OrderState.REJECTED},
      OrderState.PARTIALLY_FILLED: {OrderState.CANCELLED, OrderState.FILLED},
      OrderState.FILLED:           set(),
      OrderState.CANCELLED:        set(),
      OrderState.REJECTED:         set(),
  }
    
    
@dataclass(slots=True)
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    timestamp: datetime
    _filled_quantity: float = 0.0
    _state: OrderState = OrderState.PENDING
    
    
    def transition_to(self, new_state: OrderState) -> None:
        if new_state not in _LEGAL_TRANSITIONS[self._state]:
            raise InvalidStateTransitionError(f"{new_state} is not a legal transition from {self._state}")
        self._state = new_state
        
        
    
