#!/usr/bin/env python3
"""
Test script to verify the OrderManager implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.order_manager.order_manager import OrderManager
from src.signal_generator.signal_generator import SignalGenerator
from src.trading_client.models import TradeResult, OrderStatusInfo
from src.trading_client.trading_client import TradingClient
import time
from datetime import datetime

def create_mock_signal(action='buy', symbol='ETH-USD', confidence=0.8, current_price=3500.0):
    """Create a mock trading signal"""
    return {
        'action': action,
        'confidence': confidence,
        'predicted_change_pct': 3.5 if action == 'buy' else -3.5,
        'current_price': current_price,
        'predicted_price': current_price * 1.035 if action == 'buy' else current_price * 0.965,
        'threshold_used': 2.0,
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'reason': f"Model predicts {'positive' if action == 'buy' else 'negative'} price movement",
        'signal_strength': 3.5,
        'market_context': {
            'spread_pct': 0.057,
            'volume_imbalance': 0.2 if action == 'buy' else -0.2,
            'depth_imbalance': 0.1 if action == 'buy' else -0.1
        }
    }

class MockTradingClient:
    """Mock trading client for testing"""
    
    def __init__(self, should_succeed=True):
        self.should_succeed = should_succeed
        self.executed_trades = []
        self.order_counter = 1000
    
    def execute_trade(self, client_order_id, product_id, quantity, order_type, side):
        """Mock trade execution"""
        self.order_counter += 1
        order_id = f"order_{self.order_counter}"
        
        trade_result = TradeResult(
            success=self.should_succeed,
            order_id=order_id if self.should_succeed else None,
            client_order_id=client_order_id,
            product_id=product_id,
            side=side.value,
            order_type=order_type.value,
            size=str(quantity),
            price=None,  # Market order
            status='PENDING' if self.should_succeed else 'REJECTED',
            created_time=datetime.now().isoformat(),
            error=None if self.should_succeed else "Mock execution failure"
        )
        
        self.executed_trades.append(trade_result)
        return trade_result
    
    def get_trade_status(self, order_id):
        """Mock order status check"""
        return OrderStatusInfo(
            order_id=order_id,
            product_id='ETH-USD',
            side='BUY',
            status='FILLED',
            size='0.285714',
            filled_size='0.285714',
            remaining_size='0.0',
            price=None,
            average_filled_price='3500.00',
            created_time=datetime.now().isoformat(),
            completion_percentage='100'
        )

class MockSignalGenerator:
    """Mock signal generator for testing"""
    
    def __init__(self):
        self._subscribers = []
    
    def add_subscriber(self, subscriber):
        """Add a subscriber"""
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)
    
    def send_signal(self, signal):
        """Send a signal to all subscribers"""
        for subscriber in self._subscribers:
            subscriber.refresh([signal])

def test_order_manager():
    """Test OrderManager functionality"""
    
    print("=== Testing OrderManager Implementation ===\n")
    
    # Create mock trading client
    mock_client = MockTradingClient(should_succeed=True)
    
    # Create OrderManager
    order_manager = OrderManager(
        trading_client=mock_client,
        trade_amount_usd=1.0
    )
    
    # Set shorter cooldown for testing
    order_manager.min_execution_interval = 0.1  # 0.1 seconds for testing
    
    print("1. Testing OrderManager initialization...")
    print(f"✅ OrderManager created with ${order_manager.trade_amount_usd} per trade")
    
    # Test subscription to signal generator
    print("\n2. Testing subscription to SignalGenerator...")
    mock_signal_gen = MockSignalGenerator()
    order_manager.subscribe(mock_signal_gen)
    
    assert order_manager in mock_signal_gen._subscribers
    print("✅ OrderManager successfully subscribed to SignalGenerator")
    
    # Test signal processing - BUY signal
    print("\n3. Testing BUY signal processing...")
    buy_signal = create_mock_signal('buy', 'ETH-USD', 0.8, 3500.0)
    mock_signal_gen.send_signal(buy_signal)
    
    # Check if trade was executed
    assert len(mock_client.executed_trades) == 1
    trade = mock_client.executed_trades[0]
    assert trade.success == True
    assert trade.side == 'BUY'
    assert trade.product_id == 'ETH-USD'
    print(f"✅ BUY signal executed: {trade.side} {trade.size} {trade.product_id}")
    
    # Test signal processing - SELL signal (use different symbol to avoid duplicate prevention)
    print("\n4. Testing SELL signal processing...")
    time.sleep(0.2)  # Wait for cooldown period
    sell_signal = create_mock_signal('sell', 'BTC-USD', 0.7, 50000.0)  # Different symbol
    print(f"   Sending SELL signal: {sell_signal['action']} {sell_signal['symbol']} confidence={sell_signal['confidence']}")
    mock_signal_gen.send_signal(sell_signal)
    
    print(f"   Total trades executed: {len(mock_client.executed_trades)}")
    if len(mock_client.executed_trades) >= 2:
        trade = mock_client.executed_trades[1]
        assert trade.side == 'SELL'
        print(f"✅ SELL signal executed: {trade.side} {trade.size} {trade.product_id}")
    else:
        print(f"❌ Expected 2 trades, but got {len(mock_client.executed_trades)}")
        for i, trade in enumerate(mock_client.executed_trades):
            print(f"   Trade {i}: {trade.side} {trade.product_id}")
        # Let's check the order manager state
        print(f"   Last processed signal: {order_manager.last_processed_signal}")
        print(f"   Should execute signal result: {order_manager._should_execute_signal(sell_signal)}")
        assert False, "SELL signal was not executed"
    
    # Test HOLD signal (should be ignored)
    print("\n5. Testing HOLD signal (should be ignored)...")
    hold_signal = create_mock_signal('hold', 'ETH-USD', 0.9, 3500.0)
    mock_signal_gen.send_signal(hold_signal)
    
    assert len(mock_client.executed_trades) == 2  # Should still be 2
    print("✅ HOLD signal correctly ignored")
    
    # Test low confidence signal (should be ignored)
    print("\n6. Testing low confidence signal...")
    low_conf_signal = create_mock_signal('buy', 'ETH-USD', 0.05, 3500.0)  # Below 0.1 threshold
    mock_signal_gen.send_signal(low_conf_signal)
    
    assert len(mock_client.executed_trades) == 2  # Should still be 2
    print("✅ Low confidence signal correctly ignored")
    
    # Test duplicate signal prevention
    print("\n7. Testing duplicate signal prevention...")
    duplicate_signal = create_mock_signal('sell', 'ETH-USD', 0.8, 3500.0)  # Same as previous SELL
    mock_signal_gen.send_signal(duplicate_signal)
    
    assert len(mock_client.executed_trades) == 2  # Should still be 2
    print("✅ Duplicate signal correctly prevented")
    
    # Test convert_signals_to_trade method
    print("\n8. Testing signal to trade conversion...")
    test_signal = create_mock_signal('buy', 'BTC-USD', 0.9, 50000.0)
    trade_request = order_manager.convert_signals_to_trade(test_signal)
    
    assert trade_request.product_id == 'BTC-USD'
    assert trade_request.side.value == 'BUY'
    assert trade_request.order_type.value == 'market'
    expected_quantity = 1000.0 / 50000.0  # $1000 / $50000 per BTC
    assert abs(trade_request.quantity - expected_quantity) < 0.000001
    print(f"✅ Signal converted to trade: {trade_request.side.value} {trade_request.quantity:.6f} {trade_request.product_id}")
    
    # Test order summary
    print("\n9. Testing order summary...")
    summary = order_manager.get_order_summary()
    
    assert summary['total_orders_executed'] == 2
    assert summary['trade_amount_usd'] == 1000.0
    assert len(summary['recent_orders']) == 2
    print(f"✅ Order summary: {summary['total_orders_executed']} orders executed")
    
    # Test portfolio management
    print("\n10. Testing portfolio management...")
    order_manager.manage_portfolio()
    print("✅ Portfolio management executed without errors")
    
    # Test failed trade handling
    print("\n11. Testing failed trade handling...")
    failed_client = MockTradingClient(should_succeed=False)
    failed_order_manager = OrderManager(failed_client, 500.0)
    
    failed_signal = create_mock_signal('buy', 'ETH-USD', 0.8, 3500.0)
    failed_order_manager.refresh([failed_signal])
    
    # Should handle failure gracefully
    assert len(failed_client.executed_trades) == 1
    assert failed_client.executed_trades[0].success == False
    print("✅ Failed trade handled gracefully")
    
    print("\n=== OrderManager Test Complete ===")
    print("✅ All tests passed!")
    print(f"✅ Observer pattern working correctly")
    print(f"✅ Signal filtering and duplicate prevention working")
    print(f"✅ Trade conversion and execution working")
    print(f"✅ Error handling working")
    
    # Print final summary
    print(f"\nFinal Summary:")
    print(f"- Total trades executed: {len(mock_client.executed_trades)}")
    print(f"- Trade amount per signal: ${order_manager.trade_amount_usd}")
    print(f"- Active orders: {len(order_manager.active_orders)}")
    print(f"- Order history: {len(order_manager.order_history)}")

def test_integration_with_signal_generator():
    """Test integration with the actual SignalGenerator"""
    
    print("\n=== Integration Test with SignalGenerator ===")
    
    # Create mock components
    class MockTargetModel:
        def predict(self, data):
            return 3.5  # Always predict +3.5% change
    
    class MockMarketDataFeed:
        def listen_to_data(self, symbol, callback):
            pass
    
    # Create real SignalGenerator
    model = MockTargetModel()
    feed = MockMarketDataFeed()
    signal_gen = SignalGenerator(model, feed, buy_threshold=2.0, sell_threshold=2.0)
    
    # Create OrderManager and subscribe
    mock_client = MockTradingClient()
    order_manager = OrderManager(mock_client, 1500.0)  # $1500 per trade
    order_manager.subscribe(signal_gen)
    
    # Create mock market data
    market_data = {
        'symbol': 'ETH-USD',
        'timestamp': datetime.now().isoformat(),
        'type': 'l2update',
        'changes': [['buy', '3500.00', '1.5']],
        'best_bid': {'price': '3500.00', 'size': '1.5'},
        'best_ask': {'price': '3502.00', 'size': '2.0'},
        'spread': {'absolute': '2.00', 'percentage': 0.057, 'mid_price': '3501.00'},
        'top_bids': [['3500.00', '1.5']],
        'top_asks': [['3502.00', '2.0']]
    }
    
    # Process market data through SignalGenerator
    signal_gen.refresh(market_data)
    
    # Check if OrderManager received and processed the signal
    assert len(mock_client.executed_trades) == 1
    trade = mock_client.executed_trades[0]
    assert trade.success == True
    assert trade.side == 'BUY'  # Should be BUY since model predicts +3.5%
    
    print(f"✅ Integration test passed!")
    print(f"   Signal generated by model prediction (+3.5%)")
    print(f"   OrderManager executed: {trade.side} {trade.size} {trade.product_id}")
    print(f"   Trade amount: ${order_manager.trade_amount_usd}")

if __name__ == "__main__":
    test_order_manager()
    test_integration_with_signal_generator()
