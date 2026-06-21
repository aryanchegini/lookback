from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True,slots=True)
class Bar: 
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    
bar = Bar(datetime(2024, 1, 2), "AAPL", 1.0, 2.0, 0.5, 1.5, 100.0)
print(bar.__slots__)
    
    
