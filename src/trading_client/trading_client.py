"""
TradingClient with improved OOP design patterns
"""

from typing import Optional, Union
from coinbase.rest import RESTClient

from .models import (
    TradeRequest, TradeResult, OrderStatusInfo, CancelResult, 
    AccountBalance, Balance, OrderType, OrderSide
)
from .validators import TradeValidator, OrderValidator
from .strategies import OrderStrategyFactory
from .client_factory import CoinbaseClientFactory
from .exceptions import ValidationError, OrderError, APIError, AuthenticationError


class TradingClient:
    """
    Main trading client with improved OOP design patterns.
    
    Uses Strategy pattern for order types, Factory pattern for client creation,
    and proper separation of concerns with validators and models.
    """
    
    def __init__(self, coinbase_client: Optional[RESTClient] = None,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 environment: Optional[str] = None):
        """
        Initialize the TradingClient.
        
        Args:
            coinbase_client: Pre-configured Coinbase client (optional)
            api_key: API key (optional, loads from environment if not provided)
            api_secret: API secret (optional, loads from environment if not provided)
            environment: Environment ('sandbox' or 'production', optional)
        """
        if coinbase_client is not None:
            self._client = coinbase_client
        else:
            self._client = CoinbaseClientFactory.create_client(
                api_key=api_key,
                api_secret=api_secret,
                environment=environment
            )
        
        self._strategy_factory = OrderStrategyFactory()
    
    def execute_trade(self, 
                     product_id: str, 
                     quantity: float, 
                     price: Optional[float] = None, 
                     order_type: Union[str, OrderType] = OrderType.MARKET, 
                     side: Optional[Union[str, OrderSide]] = None) -> TradeResult:
        """
        Execute a trade order.
        
        Args:
            product_id: Trading pair (e.g., 'BTC-USD', 'ETH-USD')
            quantity: Amount to trade (positive number)
            price: Price for limit orders (required for limit orders)
            order_type: Order type ('market' or 'limit')
            side: Order side ('buy' or 'sell', auto-determined if not provided)
            
        Returns:
            TradeResult with execution details
            
        Raises:
            ValidationError: If input parameters are invalid
            OrderError: If order execution fails
        """
        try:
            # Validate and convert parameters
            TradeValidator.validate_product_id(product_id)
            TradeValidator.validate_quantity(quantity)
            
            validated_order_type = TradeValidator.validate_order_type(order_type)
            validated_side = TradeValidator.validate_side(side)
            
            TradeValidator.validate_price(price, validated_order_type)
            
            # Determine side if not provided
            if validated_side is None:
                validated_side = TradeValidator.determine_side_from_quantity(quantity)
            
            # Create trade request
            trade_request = TradeRequest(
                product_id=product_id,
                quantity=abs(quantity),  # Ensure positive quantity
                order_type=validated_order_type,
                side=validated_side,
                price=price
            )
            
            # Get strategy and prepare order parameters
            strategy = self._strategy_factory.create_strategy(validated_order_type)
            order_params = strategy.prepare_order_params(trade_request)
            
            # Execute the order
            response = self._client.create_order(**order_params)
            
            return self._process_trade_response(response, strategy.get_order_type_name(), price)
            
        except ValidationError:
            raise
        except Exception as e:
            raise OrderError(f"Failed to execute trade: {str(e)}")
    
    def cancel_trade(self, order_id: str) -> CancelResult:
        """
        Cancel a pending order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            CancelResult with cancellation details
            
        Raises:
            ValidationError: If order_id is invalid
            OrderError: If cancellation fails
        """
        try:
            OrderValidator.validate_order_id(order_id)
            
            response = self._client.cancel_orders(order_ids=[order_id])
            
            return self._process_cancel_response(response, order_id)
            
        except ValidationError:
            raise
        except Exception as e:
            raise OrderError(f"Failed to cancel trade: {str(e)}")
    
    def get_trade_status(self, order_id: str) -> OrderStatusInfo:
        """
        Get order status and details.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            OrderStatusInfo with detailed information
            
        Raises:
            ValidationError: If order_id is invalid
            OrderError: If status retrieval fails
        """
        try:
            OrderValidator.validate_order_id(order_id)
            
            response = self._client.get_order(order_id)
            
            return self._process_status_response(response, order_id)
            
        except ValidationError:
            raise
        except Exception as e:
            raise OrderError(f"Failed to get trade status: {str(e)}")
    
    def get_account_balance(self) -> AccountBalance:
        """
        Get account balances for all currencies.
        
        Returns:
            AccountBalance with balance information
            
        Raises:
            APIError: If balance retrieval fails
        """
        try:
            response = self._client.get_accounts()
            
            return self._process_balance_response(response)
            
        except Exception as e:
            raise APIError(f"Failed to retrieve account balance: {str(e)}")
    
    def _process_trade_response(self, response: dict, order_type: str, price: Optional[float]) -> TradeResult:
        """Process trade execution response"""
        if response.get('success'):
            order_data = response.get('order', {})
            order_config = order_data.get('order_configuration', {})
            
            # Extract size from appropriate configuration
            size = (order_config.get('market_market_ioc', {}).get('base_size') or 
                   order_config.get('limit_limit_gtc', {}).get('base_size') or
                   order_config.get('market_market_ioc', {}).get('quote_size'))
            
            return TradeResult(
                success=True,
                order_id=order_data.get('order_id'),
                product_id=order_data.get('product_id'),
                side=order_data.get('side'),
                order_type=order_type,
                size=size,
                price=price,
                status=order_data.get('status'),
                created_time=order_data.get('created_time'),
                response=response
            )
        else:
            error_message = response.get('error_response', {}).get('message', 'Unknown error')
            return TradeResult(
                success=False,
                error=error_message,
                response=response
            )
    
    def _process_cancel_response(self, response: dict, order_id: str) -> CancelResult:
        """Process order cancellation response"""
        if response.get('results'):
            result = response['results'][0]
            success = result.get('success', False)
            
            return CancelResult(
                success=success,
                order_id=order_id,
                status='cancelled' if success else 'failed',
                message=result.get('failure_reason') if not success else 'Order cancelled successfully',
                response=response
            )
        else:
            return CancelResult(
                success=False,
                order_id=order_id,
                status='failed',
                message='No results returned from cancel request',
                response=response
            )
    
    def _process_status_response(self, response: dict, order_id: str) -> OrderStatusInfo:
        """Process order status response"""
        if response.get('order'):
            order = response['order']
            order_config = order.get('order_configuration', {})
            market_config = order_config.get('market_market_ioc', {})
            limit_config = order_config.get('limit_limit_gtc', {})
            
            size = (market_config.get('base_size') or 
                   limit_config.get('base_size') or 
                   market_config.get('quote_size'))
            
            filled_size = order.get('filled_size', '0')
            remaining_size = str(max(0, float(size or 0) - float(filled_size)))
            
            return OrderStatusInfo(
                order_id=order.get('order_id'),
                product_id=order.get('product_id'),
                side=order.get('side'),
                status=order.get('status'),
                size=size,
                filled_size=filled_size,
                remaining_size=remaining_size,
                price=limit_config.get('limit_price'),
                average_filled_price=order.get('average_filled_price'),
                created_time=order.get('created_time'),
                completion_percentage=order.get('completion_percentage', '0'),
                time_in_force=order.get('time_in_force'),
                response=response
            )
        else:
            raise OrderError(f"Order not found: {order_id}")
    
    def _process_balance_response(self, response: dict) -> AccountBalance:
        """Process account balance response"""
        balances = {}
        total_value_usd = 0.0
        
        for account in response.get('accounts', []):
            currency = account.get('currency')
            available_balance = float(account.get('available_balance', {}).get('value', 0))
            hold_balance = float(account.get('hold', {}).get('value', 0))
            total_balance = available_balance + hold_balance
            
            if total_balance > 0 or currency in ['USD', 'USDC']:
                balance = Balance(
                    currency=currency,
                    balance=total_balance,
                    available=available_balance,
                    hold=hold_balance
                )
                balances[currency] = balance
                
                # Add to total USD value (simplified)
                if currency in ['USD', 'USDC']:
                    total_value_usd += total_balance
        
        return AccountBalance(
            balances=balances,
            total_value_usd=total_value_usd,
            timestamp=response.get('timestamp')
        )
    
    # Legacy methods (placeholders for future implementation)
    def get_portfolio(self):
        """Get portfolio holdings (not yet implemented)"""
        raise NotImplementedError("get_portfolio is not yet implemented")
    
    def view_position(self, product_id: str):
        """View position for a specific product (not yet implemented)"""
        raise NotImplementedError("view_position is not yet implemented")
