"""
Data models for TradingClient
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class TradeRequest:
    """Represents a trade request"""
    product_id: str
    quantity: float
    order_type: OrderType = OrderType.MARKET
    side: Optional[OrderSide] = None
    price: Optional[float] = None
    
    def __post_init__(self):
        """Validate the trade request after initialization"""
        if not self.product_id:
            raise ValueError("product_id is required")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("price is required for limit orders")
        if self.order_type == OrderType.LIMIT and self.price <= 0:
            raise ValueError("price must be positive for limit orders")

@dataclass
class TradeResult:
    """Represents the result of a trade execution"""
    success: bool
    order_id: Optional[str] = None
    product_id: Optional[str] = None
    side: Optional[str] = None
    order_type: Optional[str] = None
    size: Optional[str] = None
    price: Optional[float] = None
    status: Optional[str] = None
    created_time: Optional[str] = None
    error: Optional[str] = None
    response: Optional[Dict[str, Any]] = None

@dataclass
class OrderStatusInfo:
    """Represents order status information"""
    order_id: str
    product_id: str
    side: str
    status: str
    size: str
    filled_size: str
    remaining_size: str
    price: Optional[str] = None
    average_filled_price: Optional[str] = None
    created_time: Optional[str] = None
    completion_percentage: str = "0"
    time_in_force: Optional[str] = None
    response: Optional[Dict[str, Any]] = None

@dataclass
class CancelResult:
    """Represents the result of order cancellation"""
    success: bool
    order_id: str
    status: str
    message: str
    response: Optional[Dict[str, Any]] = None

@dataclass
class Balance:
    """Represents account balance for a currency"""
    currency: str
    balance: float
    available: float
    hold: float

@dataclass
class AccountBalance:
    """Represents complete account balance information"""
    balances: Dict[str, Balance]
    total_value_usd: float
    timestamp: Optional[str] = None
