#!/usr/bin/env python3
"""
Demonstration script for SignalGenerator functionality
Shows how to use the SignalGenerator with mock data and real components
"""

import sys
import time
import json
from typing import Dict, List

# Add src directory to path
sys.path.append('src')

from signal_generator.signal_generator import SignalGenerator
from models.target_model import TargetModel
from market_feed.market_feed import MarketDataFeed
from order_manager.order_manager import OrderManager


class DemoTargetModel(TargetModel):
    """Demo implementation of TargetModel for demonstration"""
    
    def predict(self, data: Dict) -> List[Dict]:
        """
        Simple demo prediction logic based on market features
        """
        signals = []
        
        # Get key market metrics
        bid_price = data.get('bid_price', 0)
        ask_price = data.get('ask_price', 0)
        volume_imbalance = data.get('volume_imbalance', 0)
        spread_percentage = data.get('spread_percentage', 0)
        depth_imbalance = data.get('depth_imbalance', 0)
        
        # Simple trading logic for demonstration
        if volume_imbalance > 0.3 and depth_imbalance > 0.2:
            # Strong buy pressure
            signals.append({
                'action': 'buy',
                'confidence': 0.8,
                'price': bid_price,
                'size': 1.0,
                'reason': 'Strong buy pressure detected'
            })
        elif volume_imbalance < -0.3 and depth_imbalance < -0.2:
            # Strong sell pressure
            signals.append({
                'action': 'sell',
                'confidence': 0.8,
                'price': ask_price,
                'size': 1.0,
                'reason': 'Strong sell pressure detected'
            })
        elif spread_percentage < 0.05:
            # Tight spread, good for market making
            signals.append({
                'action': 'market_make',
                'confidence': 0.6,
                'bid_price': bid_price,
                'ask_price': ask_price,
                'size': 0.5,
                'reason': 'Tight spread opportunity'
            })
        else:
            # Hold position
            signals.append({
                'action': 'hold',
                'confidence': 0.4,
                'reason': 'No clear signal'
            })
        
        return signals
    
    def train(self, data, target):
        print("Training model with demo data...")
    
    def evaluate(self, data, target):
        return {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.78}
    
    def save_model(self, filepath):
        print(f"Saving model to {filepath}")


class DemoOrderManager(OrderManager):
    """Demo OrderManager that prints received signals"""
    
    def __init__(self, name: str):
        super().__init__(None)  # No trading client for demo
        self.name = name
        self.received_signals = []
    
    def refresh(self, signals: List[Dict]):
        """Receive and process signals from SignalGenerator"""
        self.received_signals.extend(signals)
        
        print(f"\n📨 {self.name} received {len(signals)} signals:")
        for i, signal in enumerate(signals, 1):
            print(f"  {i}. Action: {signal.get('action', 'unknown')}")
            print(f"     Confidence: {signal.get('confidence', 0):.2f}")
            if 'reason' in signal:
                print(f"     Reason: {signal['reason']}")
            if 'price' in signal:
                print(f"     Price: ${signal['price']:.2f}")
            print()


def create_sample_market_data(symbol: str = 'ETH-USD', scenario: str = 'normal') -> Dict:
    """Create sample market data for different scenarios"""
    
    base_data = {
        'symbol': symbol,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        'type': 'l2update'
    }
    
    if scenario == 'buy_pressure':
        # Scenario with strong buy pressure
        return {
            **base_data,
            'changes': [
                ['buy', '2500.00', '5.0'],   # Large buy order
                ['buy', '2499.50', '3.0'],   # Another buy
                ['sell', '2501.00', '0.5']   # Small sell
            ],
            'best_bid': {'price': '2500.00', 'size': '5.0'},
            'best_ask': {'price': '2501.00', 'size': '1.0'},
            'spread': {'absolute': '1.00', 'percentage': 0.04, 'mid_price': '2500.50'},
            'top_bids': [
                ['2500.00', '5.0'],
                ['2499.50', '3.0'],
                ['2499.00', '2.0']
            ],
            'top_asks': [
                ['2501.00', '1.0'],
                ['2501.50', '0.8'],
                ['2502.00', '1.2']
            ]
        }
    
    elif scenario == 'sell_pressure':
        # Scenario with strong sell pressure
        return {
            **base_data,
            'changes': [
                ['sell', '2501.00', '4.0'],  # Large sell order
                ['sell', '2501.50', '2.5'],  # Another sell
                ['buy', '2500.00', '0.8']    # Small buy
            ],
            'best_bid': {'price': '2500.00', 'size': '1.2'},
            'best_ask': {'price': '2501.00', 'size': '4.0'},
            'spread': {'absolute': '1.00', 'percentage': 0.04, 'mid_price': '2500.50'},
            'top_bids': [
                ['2500.00', '1.2'],
                ['2499.50', '1.0'],
                ['2499.00', '0.8']
            ],
            'top_asks': [
                ['2501.00', '4.0'],
                ['2501.50', '2.5'],
                ['2502.00', '3.0']
            ]
        }
    
    elif scenario == 'tight_spread':
        # Scenario with tight spread
        return {
            **base_data,
            'changes': [
                ['buy', '2500.40', '1.5'],
                ['sell', '2500.50', '1.8']
            ],
            'best_bid': {'price': '2500.40', 'size': '1.5'},
            'best_ask': {'price': '2500.50', 'size': '1.8'},
            'spread': {'absolute': '0.10', 'percentage': 0.004, 'mid_price': '2500.45'},  # Very tight spread
            'top_bids': [
                ['2500.40', '1.5'],
                ['2500.30', '2.0'],
                ['2500.20', '1.8']
            ],
            'top_asks': [
                ['2500.50', '1.8'],
                ['2500.60', '2.2'],
                ['2500.70', '1.5']
            ]
        }
    
    else:  # normal scenario
        return {
            **base_data,
            'changes': [
                ['buy', '2500.00', '1.5'],
                ['sell', '2501.00', '2.0']
            ],
            'best_bid': {'price': '2500.00', 'size': '1.5'},
            'best_ask': {'price': '2501.00', 'size': '2.0'},
            'spread': {'absolute': '1.00', 'percentage': 0.04, 'mid_price': '2500.50'},
            'top_bids': [
                ['2500.00', '1.5'],
                ['2499.50', '2.0'],
                ['2499.00', '1.8']
            ],
            'top_asks': [
                ['2501.00', '2.0'],
                ['2501.50', '1.8'],
                ['2502.00', '2.2']
            ]
        }


def demo_signal_generator():
    """Demonstrate SignalGenerator functionality"""
    
    print("🚀 SignalGenerator Demonstration")
    print("=" * 60)
    
    # Create components
    print("1. Creating components...")
    target_model = DemoTargetModel()
    market_feed = MarketDataFeed()  # Mock - won't actually connect
    
    # Create order managers (subscribers)
    order_manager1 = DemoOrderManager("Portfolio Manager")
    order_manager2 = DemoOrderManager("Risk Manager")
    
    # Create SignalGenerator
    signal_gen = SignalGenerator(
        target_model=target_model,
        market_data_feed=market_feed,
        subscribers=[order_manager1, order_manager2]
    )
    
    print("✅ Components created successfully")
    
    # Test different market scenarios
    scenarios = [
        ('normal', 'Normal Market Conditions'),
        ('buy_pressure', 'Strong Buy Pressure'),
        ('sell_pressure', 'Strong Sell Pressure'),
        ('tight_spread', 'Tight Spread Opportunity')
    ]
    
    print(f"\n2. Testing {len(scenarios)} market scenarios...")
    
    for scenario_key, scenario_name in scenarios:
        print(f"\n{'='*60}")
        print(f"📊 Scenario: {scenario_name}")
        print('='*60)
        
        # Create sample market data
        market_data = create_sample_market_data('ETH-USD', scenario_key)
        
        # Show market data summary
        print("📈 Market Data:")
        print(f"  Symbol: {market_data['symbol']}")
        print(f"  Best Bid: ${market_data['best_bid']['price']} (Size: {market_data['best_bid']['size']})")
        print(f"  Best Ask: ${market_data['best_ask']['price']} (Size: {market_data['best_ask']['size']})")
        print(f"  Spread: {market_data['spread']['percentage']:.3f}%")
        print(f"  Changes: {len(market_data['changes'])} order updates")
        
        # Process market data through SignalGenerator
        signal_gen._market_data_callback(market_data)
        
        # Show extracted features
        features = signal_gen._extract_market_features(market_data)
        print(f"\n🔍 Extracted Features:")
        print(f"  Volume Imbalance: {features['volume_imbalance']:.3f}")
        print(f"  Depth Imbalance: {features['depth_imbalance']:.3f}")
        print(f"  Total Volume: {features['total_volume']:.2f}")
        print(f"  Mid Price: ${features['mid_price']:.2f}")
        
        # Show generated signals
        print(f"\n🎯 Generated Signals: {len(signal_gen.signals)}")
        for i, signal in enumerate(signal_gen.signals, 1):
            print(f"  Signal {i}: {signal}")
        
        time.sleep(1)  # Brief pause between scenarios
    
    # Show market summary
    print(f"\n{'='*60}")
    print("📋 Final Market Summary")
    print('='*60)
    
    summary = signal_gen.get_market_summary()
    if summary:
        print(f"Symbol: {summary['symbol']}")
        print(f"Last Update: {summary['last_update']}")
        print(f"Best Bid: ${summary['best_bid']['price']}")
        print(f"Best Ask: ${summary['best_ask']['price']}")
        print(f"Price Trend: {summary['price_trend']}")
        print(f"Recent Signals: {summary['recent_signals']}")
    
    # Show subscriber statistics
    print(f"\n📊 Subscriber Statistics:")
    print(f"  {order_manager1.name}: {len(order_manager1.received_signals)} signals received")
    print(f"  {order_manager2.name}: {len(order_manager2.received_signals)} signals received")
    
    print(f"\n✅ Demonstration completed successfully!")
    print("=" * 60)


def demo_feature_extraction():
    """Demonstrate feature extraction capabilities"""
    
    print("\n🔬 Feature Extraction Demonstration")
    print("=" * 60)
    
    # Create SignalGenerator for feature extraction
    target_model = DemoTargetModel()
    market_feed = MarketDataFeed()
    signal_gen = SignalGenerator(target_model, market_feed)
    
    # Test order flow analysis
    print("1. Order Flow Analysis:")
    changes = [
        ['buy', '2500.00', '2.5'],
        ['buy', '2499.50', '1.8'],
        ['sell', '2501.00', '1.2'],
        ['sell', '2501.50', '3.0'],
        ['buy', '2499.00', '0']  # Remove order
    ]
    
    flow_metrics = signal_gen._analyze_order_flow(changes)
    print(f"  Buy Volume: {flow_metrics['buy_volume']:.2f}")
    print(f"  Sell Volume: {flow_metrics['sell_volume']:.2f}")
    print(f"  Volume Imbalance: {flow_metrics['volume_imbalance']:.3f}")
    print(f"  Order Imbalance: {flow_metrics['order_imbalance']:.3f}")
    
    # Test order book depth analysis
    print("\n2. Order Book Depth Analysis:")
    bids = [['2500.00', '2.5'], ['2499.50', '1.8'], ['2499.00', '3.2']]
    asks = [['2501.00', '1.2'], ['2501.50', '3.0'], ['2502.00', '2.1']]
    
    depth_metrics = signal_gen._analyze_order_book_depth(bids, asks)
    print(f"  Bid Depth: {depth_metrics['bid_depth']:.2f}")
    print(f"  Ask Depth: {depth_metrics['ask_depth']:.2f}")
    print(f"  Depth Imbalance: {depth_metrics['depth_imbalance']:.3f}")
    print(f"  Weighted Mid Price: ${depth_metrics['weighted_mid_price']:.2f}")
    
    print("✅ Feature extraction demonstration completed")


if __name__ == '__main__':
    print("SignalGenerator Comprehensive Demo")
    print("=" * 60)
    
    try:
        # Run main demonstration
        demo_signal_generator()
        
        # Run feature extraction demo
        demo_feature_extraction()
        
        print(f"\n🎉 All demonstrations completed successfully!")
        print("\n📝 Key Takeaways:")
        print("  • SignalGenerator processes Level 2 market data effectively")
        print("  • Feature extraction captures order flow and depth metrics")
        print("  • Signal generation adapts to different market conditions")
        print("  • Publisher-subscriber pattern works for multiple order managers")
        print("  • Error handling ensures robust operation")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    exit(0)
