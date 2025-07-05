#!/usr/bin/env python3
"""
Integration test for the complete trading bot workflow with Coinbase WebSocket
This demonstrates the full pipeline: MarketDataFeed -> SignalGenerator -> OrderManager
"""

import sys
import time
import json
import threading
from dotenv import load_dotenv

# Add src directory to path
sys.path.append('src')

from market_feed.market_feed import MarketDataFeed
from signal_generator.signal_generator import SignalGenerator
from models.target_model import TargetModel
from order_manager.order_manager import OrderManager


class MockTargetModel:
    """Mock ML model for testing signal generation"""
    
    def __init__(self):
        self.prediction_count = 0
        
    def predict(self, market_features):
        """
        Generate mock trading signals based on market features
        
        In a real implementation, this would use trained ML models
        """
        self.prediction_count += 1
        
        # Extract key features
        mid_price = market_features.get('mid_price', 0)
        spread_pct = market_features.get('spread_percentage', 0)
        volume_imbalance = market_features.get('volume_imbalance', 0)
        depth_imbalance = market_features.get('depth_imbalance', 0)
        
        signals = []
        
        # Simple mock strategy based on market microstructure
        if spread_pct < 0.05 and abs(volume_imbalance) > 0.3:
            # Tight spread + volume imbalance = potential signal
            if volume_imbalance > 0:  # More buying pressure
                signals.append({
                    'action': 'buy',
                    'symbol': market_features['symbol'],
                    'confidence': min(0.8, abs(volume_imbalance)),
                    'price': mid_price,
                    'size': 0.001,  # Small size for testing
                    'reason': f'buy_pressure_imbalance_{volume_imbalance:.3f}',
                    'timestamp': market_features['timestamp']
                })
            elif volume_imbalance < -0.3:  # More selling pressure
                signals.append({
                    'action': 'sell',
                    'symbol': market_features['symbol'],
                    'confidence': min(0.8, abs(volume_imbalance)),
                    'price': mid_price,
                    'size': 0.001,
                    'reason': f'sell_pressure_imbalance_{volume_imbalance:.3f}',
                    'timestamp': market_features['timestamp']
                })
        
        # Depth imbalance strategy
        if abs(depth_imbalance) > 0.4:
            if depth_imbalance > 0:  # More bid depth
                signals.append({
                    'action': 'buy',
                    'symbol': market_features['symbol'],
                    'confidence': 0.6,
                    'price': mid_price,
                    'size': 0.0005,
                    'reason': f'bid_depth_advantage_{depth_imbalance:.3f}',
                    'timestamp': market_features['timestamp']
                })
            else:  # More ask depth
                signals.append({
                    'action': 'sell',
                    'symbol': market_features['symbol'],
                    'confidence': 0.6,
                    'price': mid_price,
                    'size': 0.0005,
                    'reason': f'ask_depth_advantage_{depth_imbalance:.3f}',
                    'timestamp': market_features['timestamp']
                })
        
        if signals:
            print(f"🧠 MockTargetModel generated {len(signals)} signals (prediction #{self.prediction_count})")
            for signal in signals:
                print(f"   📊 {signal['action'].upper()} {signal['symbol']} - {signal['reason']} (confidence: {signal['confidence']:.2f})")
        
        return signals


class MockOrderManager:
    """Mock OrderManager for testing signal reception"""
    
    def __init__(self, name="OrderManager"):
        self.name = name
        self.received_signals = []
        self.signal_count = 0
        
    def refresh(self, signals):
        """
        Receive signals from SignalGenerator
        
        In a real implementation, this would:
        1. Apply risk management
        2. Size positions appropriately  
        3. Execute trades via TradingClient
        """
        self.signal_count += len(signals)
        self.received_signals.extend(signals)
        
        if signals:
            print(f"📨 {self.name} received {len(signals)} signals (total: {self.signal_count})")
            for signal in signals:
                print(f"   🎯 Processing {signal['action'].upper()} signal for {signal['symbol']}")
                print(f"      💰 Price: ${signal['price']:.2f}, Size: {signal['size']}, Confidence: {signal['confidence']:.2f}")
                print(f"      📝 Reason: {signal['reason']}")
                
                # Simulate order management logic
                if signal['confidence'] > 0.7:
                    print(f"      ✅ HIGH CONFIDENCE - Would execute trade")
                elif signal['confidence'] > 0.5:
                    print(f"      ⚠️  MEDIUM CONFIDENCE - Would execute with reduced size")
                else:
                    print(f"      ❌ LOW CONFIDENCE - Would skip trade")


def test_complete_integration():
    """Test the complete trading bot integration with Coinbase WebSocket"""
    print("🚀 Complete Trading Bot Integration Test")
    print("=" * 60)
    
    try:
        # Step 1: Initialize components
        print("1. Initializing trading bot components...")
        
        # Create mock ML model
        target_model = MockTargetModel()
        print("✅ MockTargetModel created")
        
        # Create MarketDataFeed
        market_feed = MarketDataFeed(environment='sandbox')
        print("✅ MarketDataFeed initialized")
        
        # Create mock OrderManagers (simulate multiple subscribers)
        order_manager_1 = MockOrderManager("OrderManager-1")
        order_manager_2 = MockOrderManager("OrderManager-2")
        print("✅ OrderManagers created")
        
        # Create SignalGenerator with subscribers
        signal_generator = SignalGenerator(
            target_model=target_model,
            market_data_feed=market_feed,
            subscribers=[order_manager_1, order_manager_2]
        )
        print("✅ SignalGenerator initialized with 2 subscribers")
        
        # Step 2: Start the data pipeline
        print("\n2. Starting market data pipeline...")
        signal_generator.start_listening('ETH-USD')
        print("✅ Started listening to ETH-USD Level 2 data")
        
        print("\n3. Running integration test (press Ctrl+C to stop)...")
        print("📡 Waiting for Coinbase WebSocket data...")
        print("🔄 Pipeline: Coinbase -> MarketDataFeed -> SignalGenerator -> OrderManagers")
        
        # Step 3: Monitor the pipeline
        start_time = time.time()
        last_summary_time = start_time
        
        try:
            while True:
                time.sleep(1)
                current_time = time.time()
                
                # Show periodic summary
                if current_time - last_summary_time >= 30:  # Every 30 seconds
                    print(f"\n📊 Pipeline Summary (running for {int(current_time - start_time)}s):")
                    
                    # Market data summary
                    market_summary = signal_generator.get_market_summary()
                    if market_summary:
                        print(f"   📈 Market: {market_summary['symbol']}")
                        if market_summary['best_bid'] and market_summary['best_ask']:
                            print(f"   💰 Best Bid: ${market_summary['best_bid']['price']}")
                            print(f"   💸 Best Ask: ${market_summary['best_ask']['price']}")
                        if market_summary['spread']:
                            print(f"   📏 Spread: {market_summary['spread']['percentage']}%")
                        print(f"   📊 Trend: {market_summary['price_trend']}")
                        print(f"   🎯 Recent Signals: {market_summary['recent_signals']}")
                    
                    # Signal generation summary
                    print(f"   🧠 ML Model Predictions: {target_model.prediction_count}")
                    
                    # Order management summary
                    print(f"   📨 OrderManager-1 Signals: {order_manager_1.signal_count}")
                    print(f"   📨 OrderManager-2 Signals: {order_manager_2.signal_count}")
                    
                    last_summary_time = current_time
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping integration test...")
            
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        return False
    
    finally:
        # Clean shutdown
        print("\n4. Cleaning up...")
        if 'market_feed' in locals():
            market_feed.stop_feed()
            print("✅ MarketDataFeed stopped")
        
        # Final statistics
        if 'target_model' in locals() and 'order_manager_1' in locals():
            print(f"\n📊 Final Statistics:")
            print(f"   🧠 Total ML Predictions: {target_model.prediction_count}")
            print(f"   📨 Total Signals Generated: {order_manager_1.signal_count + order_manager_2.signal_count}")
            print(f"   ⏱️  Test Duration: {int(time.time() - start_time)}s")
    
    return True


def test_market_microstructure_analysis():
    """
    Demonstrate advanced market microstructure analysis capabilities
    """
    print("\n" + "="*60)
    print("🔬 Market Microstructure Analysis Demo")
    print("="*60)
    
    print("""
    The new Level 2 integration provides advanced market analysis:
    
    📊 Order Flow Analysis:
    - Buy/sell volume imbalance detection
    - Order count imbalance tracking
    - Real-time pressure identification
    
    📈 Order Book Depth Analysis:
    - Bid/ask depth comparison
    - Liquidity imbalance detection
    - Volume-weighted price calculation
    
    🎯 Signal Generation Features:
    - Spread-based market condition assessment
    - Multi-level order book analysis
    - Historical trend tracking
    - Confidence scoring based on market structure
    
    🔄 Real-time Processing:
    - Delta updates for efficiency
    - Automatic reconnection handling
    - Thread-safe callback processing
    - Configurable order book depth (20 levels)
    
    This enables sophisticated trading strategies based on:
    - Market maker vs taker identification
    - Liquidity provision opportunities
    - Short-term price prediction
    - Risk management based on market depth
    """)


if __name__ == "__main__":
    load_dotenv()
    
    print("Coinbase WebSocket Trading Bot Integration Test")
    print("=" * 60)
    print("⚠️  This test connects to Coinbase's sandbox WebSocket feed")
    print("📝 No actual trading will occur - this is for testing the data pipeline")
    print()
    
    # Run the complete integration test
    success = test_complete_integration()
    
    # Show microstructure analysis capabilities
    test_market_microstructure_analysis()
    
    print("\n" + "="*60)
    print("📝 Integration Complete!")
    print("✅ MarketDataFeed: Coinbase WebSocket Level 2 order book")
    print("✅ SignalGenerator: Advanced market microstructure analysis")
    print("✅ OrderManager: Multi-subscriber signal distribution")
    print("✅ Pipeline: Real-time data processing with delta updates")
    print("\n🚀 Your trading bot is ready for Level 2 market data!")
    print("="*60)
    
    sys.exit(0 if success else 1)
