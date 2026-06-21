from dataclasses import dataclass
from datetime import datetime
from enum import Enum


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

    

# pending -> submitted, cancelled, rejected (pre-submit rejection, your own checks refuse sending it out to the exchange)
# submitted -> filled, partially_filled, cancelled, rejected (rejected by exchange)
# partially_filled -> cancelled, filled (partially filled to rejected is very rare)
# filled -> none
# cancelled -> none
# rejected -> none


