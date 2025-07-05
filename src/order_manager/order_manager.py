from typing import List, Dict, Optional, TYPE_CHECKING
import logging
import time
import uuid
from datetime import datetime

if TYPE_CHECKING:
    from src.trading_client.trading_client import TradingClient
    from src.signal_generator.signal_generator import SignalGenerator

from src.trading_client.models import TradeRequest, OrderType, OrderSide, TradeResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderManager:
    """
    OrderManager acts as an observer to SignalGenerator and converts trading signals
    into executable trades through the TradingClient.
    
    Key Features:
    - Observer pattern: subscribes to SignalGenerator for signal updates
    - Simple position sizing: fixed USD amount per trade
    - Market orders: immediate execution
    - Duplicate prevention: avoids executing similar signals too frequently
    """
    
    def __init__(self, trading_client: 'TradingClient', trade_amount_usd: float = 1.0):
        """
        Initialize the OrderManager with a trading client.

        Parameters:
        - trading_client: An instance of TradingClient that will be used to execute trades
        - trade_amount_usd: Fixed USD amount to trade per signal (default: $1000)

        Returns:
        - None
        """
        self.trading_client = trading_client
        self.trade_amount_usd = trade_amount_usd
        
        # Signal processing state
        self.latest_signals = []
        self.last_processed_signal = None
        self.last_execution_time = 0
        
        # Order tracking
        self.active_orders = {}  # order_id -> TradeResult
        self.order_history = []  # List of all executed orders
        
        # Configuration
        self.min_execution_interval = 60  # Minimum seconds between similar trades
        self.min_confidence_threshold = 0.1  # Minimum signal confidence to act on
        
        logger.info(f"OrderManager initialized with ${trade_amount_usd} per trade")
    
    def subscribe(self, signal_generator: 'SignalGenerator'):
        """
        Subscribe to a signal generator to receive trading signals.

        Parameters:
        - signal_generator: An instance of SignalGenerator that provides trading signals

        Returns:
        - None
        """
        if self not in signal_generator._subscribers:
            signal_generator._subscribers.append(self)
            logger.info(f"OrderManager subscribed to SignalGenerator")
        else:
            logger.warning("OrderManager already subscribed to this SignalGenerator")
    
    def refresh(self, signals: List[Dict]):
        """
        Refresh the order manager's state with new signals from SignalGenerator.
        This method is called automatically by the SignalGenerator when new signals arrive.

        Parameters:
        - signals: List of trading signal dictionaries

        Returns:
        - None
        """
        try:
            self.latest_signals = signals
            logger.debug(f"OrderManager received {len(signals)} signals")
            
            # Process the latest signals
            self.process_latest_signals()
            
        except Exception as e:
            logger.error(f"Error in OrderManager.refresh: {e}")
    
    def process_latest_signals(self):
        """
        Process the latest signals and execute trades if conditions are met.
        Only acts on buy/sell signals, ignores hold signals.
        """
        if not self.latest_signals:
            logger.debug("No signals to process")
            return
        
        # Take the first signal (most recent)
        signal = self.latest_signals[0]
        
        logger.debug(f"Processing signal: {signal['action']} for {signal['symbol']} "
                    f"with {signal['confidence']:.3f} confidence")
        
        # Only act on buy/sell signals with sufficient confidence
        if signal['action'] in ['buy', 'sell'] and signal['confidence'] >= self.min_confidence_threshold:
            
            # Check if we should execute this signal (avoid duplicates)
            if self._should_execute_signal(signal):
                logger.info(f"Executing {signal['action']} signal for {signal['symbol']}")
                
                # Convert signal to trade and execute
                trade_request = self.convert_signals_to_trade(signal)
                self.direct_client_to_trade(trade_request, signal)
                
                # Update tracking
                self.last_processed_signal = signal
                self.last_execution_time = time.time()
            else:
                logger.debug(f"Skipping signal execution (duplicate or too frequent)")
        else:
            logger.debug(f"Ignoring {signal['action']} signal (confidence: {signal['confidence']:.3f})")
    
    def convert_signals_to_trade(self, signal: Dict) -> TradeRequest:
        """
        Convert a trading signal into an executable trade order.

        Parameters:
        - signal: Trading signal dictionary from SignalGenerator

        Returns:
        - TradeRequest object that can be executed by the trading client
        """
        try:
            # Calculate quantity based on fixed USD amount and current price
            current_price = signal['current_price']
            if current_price <= 0:
                raise ValueError(f"Invalid current price: {current_price}")
            
            # For crypto, we buy/sell in base currency (e.g., ETH in ETH-USD)
            quantity = self.trade_amount_usd / current_price
            
            # Generate unique client order ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            client_order_id = f"signal_{signal['symbol']}_{signal['action']}_{timestamp}_{uuid.uuid4().hex[:8]}"
            
            # Determine order side
            side = OrderSide.BUY if signal['action'] == 'buy' else OrderSide.SELL
            
            # Create trade request
            trade_request = TradeRequest(
                client_order_id=client_order_id,
                product_id=signal['symbol'],
                quantity=quantity,
                order_type=OrderType.MARKET,  # Always use market orders for immediate execution
                side=side
            )
            
            logger.info(f"Created trade request: {side.value} {quantity:.6f} {signal['symbol']} "
                       f"at ~${current_price:.2f} (${self.trade_amount_usd} total)")
            
            return trade_request
            
        except Exception as e:
            logger.error(f"Error converting signal to trade: {e}")
            raise
    
    def direct_client_to_trade(self, trade_request: TradeRequest, original_signal: Dict):
        """
        Direct the trading client to execute the trade order.

        Parameters:
        - trade_request: TradeRequest object to execute
        - original_signal: Original signal that generated this trade (for logging)

        Returns:
        - None
        """
        try:
            logger.info(f"Executing trade: {trade_request.side.value} {trade_request.quantity:.6f} "
                       f"{trade_request.product_id}")
            
            # Execute the trade through the trading client
            result = self.trading_client.execute_trade(
                client_order_id=trade_request.client_order_id,
                product_id=trade_request.product_id,
                quantity=trade_request.quantity,
                order_type=trade_request.order_type,
                side=trade_request.side
            )
            
            # Process the result
            if result.success:
                logger.info(f"✅ Trade executed successfully: {result.order_id}")
                logger.info(f"   Order: {result.side} {result.size} {result.product_id}")
                logger.info(f"   Status: {result.status}")
                
                # Track the active order
                if result.order_id:
                    self.active_orders[result.order_id] = result
                
                # Add to order history with signal context
                order_record = {
                    'trade_result': result,
                    'original_signal': original_signal,
                    'execution_time': datetime.now().isoformat(),
                    'trade_amount_usd': self.trade_amount_usd
                }
                self.order_history.append(order_record)
                
            else:
                logger.error(f"❌ Trade execution failed: {result.error}")
                logger.error(f"   Signal: {original_signal['action']} {original_signal['symbol']}")
                
        except Exception as e:
            logger.error(f"Error directing client to trade: {e}")
            logger.error(f"Trade request: {trade_request}")
    
    def _should_execute_signal(self, signal: Dict) -> bool:
        """
        Check if we should execute this signal to avoid duplicates and over-trading.
        
        Parameters:
        - signal: Trading signal to evaluate
        
        Returns:
        - True if signal should be executed, False otherwise
        """
        # Always execute if this is the first signal
        if not self.last_processed_signal:
            return True
        
        # Check time-based cooldown
        current_time = time.time()
        time_since_last = current_time - self.last_execution_time
        
        if time_since_last < self.min_execution_interval:
            logger.debug(f"Skipping signal: {time_since_last:.1f}s since last execution "
                        f"(min: {self.min_execution_interval}s)")
            return False
        
        # Check if it's the same action for the same symbol
        same_symbol = signal['symbol'] == self.last_processed_signal['symbol']
        same_action = signal['action'] == self.last_processed_signal['action']
        
        if same_symbol and same_action:
            logger.debug(f"Skipping duplicate signal: {signal['action']} {signal['symbol']}")
            return False
        
        # Allow different symbols or different actions even within the time window
        return True
    
    def manage_portfolio(self):
        """
        Manage the portfolio by checking order statuses and updating active orders.
        This is a simple implementation that just updates order statuses.

        Returns:
        - None
        """
        if not self.active_orders:
            logger.debug("No active orders to manage")
            return
        
        logger.info(f"Managing portfolio with {len(self.active_orders)} active orders")
        
        # Check status of active orders
        completed_orders = []
        
        for order_id, trade_result in self.active_orders.items():
            try:
                # Get current order status
                status_info = self.trading_client.get_trade_status(order_id)
                
                logger.debug(f"Order {order_id}: {status_info.status} "
                           f"({status_info.filled_size}/{status_info.size} filled)")
                
                # Remove completed orders from active tracking
                if status_info.status in ['FILLED', 'CANCELLED', 'REJECTED']:
                    completed_orders.append(order_id)
                    logger.info(f"Order {order_id} completed with status: {status_info.status}")
                    
            except Exception as e:
                logger.error(f"Error checking status for order {order_id}: {e}")
        
        # Remove completed orders from active tracking
        for order_id in completed_orders:
            del self.active_orders[order_id]
    
    def get_order_summary(self) -> Dict:
        """
        Get a summary of order activity and current state.
        
        Returns:
        - Dictionary with order summary information
        """
        return {
            'active_orders': len(self.active_orders),
            'total_orders_executed': len(self.order_history),
            'trade_amount_usd': self.trade_amount_usd,
            'last_execution_time': datetime.fromtimestamp(self.last_execution_time).isoformat() if self.last_execution_time else None,
            'last_processed_signal': {
                'action': self.last_processed_signal['action'],
                'symbol': self.last_processed_signal['symbol'],
                'confidence': self.last_processed_signal['confidence']
            } if self.last_processed_signal else None,
            'recent_orders': [
                {
                    'order_id': record['trade_result'].order_id,
                    'side': record['trade_result'].side,
                    'product_id': record['trade_result'].product_id,
                    'size': record['trade_result'].size,
                    'status': record['trade_result'].status,
                    'execution_time': record['execution_time'],
                    'signal_confidence': record['original_signal']['confidence']
                }
                for record in self.order_history[-5:]  # Last 5 orders
            ]
        }
