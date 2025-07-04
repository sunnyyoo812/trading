"""
Input validation utilities for TradingClient
"""

from typing import Optional
from .models import OrderType, OrderSide
from .exceptions import ValidationError

class TradeValidator:
    """Validates trade parameters"""
    
    @staticmethod
    def validate_product_id(product_id: str) -> None:
        """Validate product ID format"""
        if not product_id:
            raise ValidationError("product_id is required")
        
        if not isinstance(product_id, str):
            raise ValidationError("product_id must be a string")
        
        # Basic format validation (e.g., BTC-USD, ETH-USD)
        if '-' not in product_id:
            raise ValidationError("product_id must be in format 'BASE-QUOTE' (e.g., 'BTC-USD')")
        
        parts = product_id.split('-')
        if len(parts) != 2:
            raise ValidationError("product_id must be in format 'BASE-QUOTE' (e.g., 'BTC-USD')")
        
        base, quote = parts
        if not base or not quote:
            raise ValidationError("Both base and quote currencies must be specified")
    
    @staticmethod
    def validate_quantity(quantity: float) -> None:
        """Validate quantity parameter"""
        if quantity is None:
            raise ValidationError("quantity is required")
        
        if not isinstance(quantity, (int, float)):
            raise ValidationError("quantity must be a number")
        
        if quantity <= 0:
            raise ValidationError("quantity must be positive")
    
    @staticmethod
    def validate_price(price: Optional[float], order_type: OrderType) -> None:
        """Validate price parameter based on order type"""
        if order_type == OrderType.LIMIT:
            if price is None:
                raise ValidationError("price is required for limit orders")
            
            if not isinstance(price, (int, float)):
                raise ValidationError("price must be a number")
            
            if price <= 0:
                raise ValidationError("price must be positive")
    
    @staticmethod
    def validate_order_type(order_type: str) -> OrderType:
        """Validate and convert order type"""
        if not order_type:
            return OrderType.MARKET
        
        if isinstance(order_type, OrderType):
            return order_type
        
        if isinstance(order_type, str):
            order_type = order_type.lower()
            if order_type == "market":
                return OrderType.MARKET
            elif order_type == "limit":
                return OrderType.LIMIT
            else:
                raise ValidationError("order_type must be 'market' or 'limit'")
        
        raise ValidationError("order_type must be a string or OrderType enum")
    
    @staticmethod
    def validate_side(side: Optional[str]) -> Optional[OrderSide]:
        """Validate and convert order side"""
        if side is None:
            return None
        
        if isinstance(side, OrderSide):
            return side
        
        if isinstance(side, str):
            side = side.lower()
            if side == "buy":
                return OrderSide.BUY
            elif side == "sell":
                return OrderSide.SELL
            else:
                raise ValidationError("side must be 'buy' or 'sell'")
        
        raise ValidationError("side must be a string or OrderSide enum")
    
    @staticmethod
    def determine_side_from_quantity(quantity: float) -> OrderSide:
        """Determine order side from quantity sign"""
        return OrderSide.BUY if quantity > 0 else OrderSide.SELL

class OrderValidator:
    """Validates order-related parameters"""
    
    @staticmethod
    def validate_order_id(order_id: str) -> None:
        """Validate order ID"""
        if not order_id:
            raise ValidationError("order_id is required")
        
        if not isinstance(order_id, str):
            raise ValidationError("order_id must be a string")
        
        if len(order_id.strip()) == 0:
            raise ValidationError("order_id cannot be empty")
