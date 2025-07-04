"""
TradingClient package with improved OOP design patterns
"""

from .trading_client import TradingClient
from .models import (
    TradeRequest, TradeResult, OrderStatusInfo, CancelResult, 
    AccountBalance, Balance, OrderType, OrderSide
)
from .exceptions import (
    TradingClientError, AuthenticationError, ValidationError, 
    OrderError, APIError
)
from .validators import TradeValidator, OrderValidator
from .strategies import OrderStrategy, OrderStrategyFactory
from .client_factory import CoinbaseClientFactory

__all__ = [
    # Main client
    'TradingClient',
    
    # Models
    'TradeRequest', 'TradeResult', 'OrderStatusInfo', 'CancelResult',
    'AccountBalance', 'Balance', 'OrderType', 'OrderSide',
    
    # Exceptions
    'TradingClientError', 'AuthenticationError', 'ValidationError',
    'OrderError', 'APIError',
    
    # Validators
    'TradeValidator', 'OrderValidator',
    
    # Strategies
    'OrderStrategy', 'OrderStrategyFactory',
    
    # Factory
    'CoinbaseClientFactory'
]

__version__ = "2.0.0"
__author__ = "Trading Bot Team"
__description__ = "Advanced TradingClient with OOP design patterns for Coinbase Advanced Trade API"
