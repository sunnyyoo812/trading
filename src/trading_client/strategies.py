"""
Strategy pattern implementations for different order types
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from .models import TradeRequest, OrderType, OrderSide

class OrderStrategy(ABC):
    """Abstract base class for order execution strategies"""
    
    @abstractmethod
    def prepare_order_params(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Prepare order parameters for API call"""
        pass
    
    @abstractmethod
    def get_order_type_name(self) -> str:
        """Get the order type name for this strategy"""
        pass

class MarketOrderStrategy(OrderStrategy):
    """Strategy for market orders"""
    
    def prepare_order_params(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Prepare market order parameters"""
        order_params = {
            'client_order_id': trade_request.client_order_id,
            'product_id': trade_request.product_id,
            'side': trade_request.side.value,
            'order_configuration': {}
        }
        
        if trade_request.side == OrderSide.BUY:
            # For market buy orders, specify quote_size (USD amount)
            order_params['order_configuration']['market_market_ioc'] = {
                'base_size': str(trade_request.quantity)
            }
        else:
            # For market sell orders, specify base_size (crypto amount)
            order_params['order_configuration']['market_market_ioc'] = {
                'base_size': str(trade_request.quantity)
            }
        
        return order_params
    
    def get_order_type_name(self) -> str:
        return "market"

class LimitOrderStrategy(OrderStrategy):
    """Strategy for limit orders"""
    
    def prepare_order_params(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Prepare limit order parameters"""
        order_params = {
            'client_order_id': trade_request.client_order_id,
            'product_id': trade_request.product_id,
            'side': trade_request.side.value,
            'order_configuration': {
                'limit_limit_gtc': {
                    'base_size': str(trade_request.quantity),
                    'limit_price': str(trade_request.price)
                }
            }
        }
        
        return order_params
    
    def get_order_type_name(self) -> str:
        return "limit"

class OrderStrategyFactory:
    """Factory for creating order strategies"""
    
    _strategies = {
        OrderType.MARKET: MarketOrderStrategy,
        OrderType.LIMIT: LimitOrderStrategy
    }
    
    @classmethod
    def create_strategy(cls, order_type: OrderType) -> OrderStrategy:
        """Create an order strategy based on order type"""
        strategy_class = cls._strategies.get(order_type)
        if not strategy_class:
            raise ValueError(f"Unsupported order type: {order_type}")
        
        return strategy_class()
    
    @classmethod
    def register_strategy(cls, order_type: OrderType, strategy_class: type) -> None:
        """Register a new order strategy"""
        cls._strategies[order_type] = strategy_class
    
    @classmethod
    def get_supported_order_types(cls) -> list:
        """Get list of supported order types"""
        return list(cls._strategies.keys())
