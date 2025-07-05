#!/usr/bin/env python3
"""
Demonstration of HeuristicModel with SignalGenerator
"""

from src.models.heuristic_model import HeuristicModel
from src.signal_generator.signal_generator import SignalGenerator
from src.order_manager.order_manager import OrderManager
from src.market_feed.market_feed import MarketDataFeed

# Mock trading client for demonstration
class MockTradingClient:
    """Mock trading client for testing"""
    
    def __init__(self, should_succeed=True):
        self.should_succeed = should_succeed
        self.executed_trades = []
        self.order_counter = 1000
    
    def execute_trade(self, client_order_id, product_id, quantity, price=None, order_type='market', side=None):
        """Mock trade execution"""
        if not self.should_succeed:
            raise Exception("Mock execution failure")
        
        self.order_counter += 1
        order_id = f"order_{self.order_counter}"
        
        trade = {
            'order_id': order_id,
            'client_order_id': client_order_id,
            'symbol': product_id,
            'side': side,
            'size': quantity,
            'price': price,
            'status': 'PENDING'
        }
        
        self.executed_trades.append(trade)
        
        # Mock successful response
        return type('MockResult', (), {
            'success': True,
            'order_id': order_id,
            'client_order_id': client_order_id,
            'product_id': product_id,
            'side': side,
            'size': str(quantity),
            'status': 'PENDING'
        })()

def demo_heuristic_model():
    """Demonstrate HeuristicModel generating random trading signals"""
    print("🎲 HeuristicModel Trading Demo")
    print("=" * 50)
    
    # 1. Create HeuristicModel
    print("1. Creating HeuristicModel...")
    model = HeuristicModel(random_seed=123)  # Different seed for variety
    print(f"   Model info: {model.get_model_info()}")
    
    # 2. Mock training (not needed but shows interface compatibility)
    print("\n2. Mock training...")
    training_data = [{'price': 100 + i, 'volume': 1000 + i*10} for i in range(5)]
    training_targets = [0.1, -0.2, 0.3, -0.1, 0.2]
    model.train(training_data, training_targets)
    
    # 3. Create OrderManager first
    print("\n3. Setting up OrderManager...")
    trading_client = MockTradingClient()
    order_manager = OrderManager(
        trading_client=trading_client,
        trade_amount_usd=500.0
    )
    
    # 4. Create SignalGenerator with HeuristicModel and OrderManager as subscriber
    print("\n4. Creating SignalGenerator with HeuristicModel...")
    market_feed = MarketDataFeed()
    signal_generator = SignalGenerator(
        target_model=model,
        market_data_feed=market_feed,
        subscribers=[order_manager],  # Pass OrderManager as subscriber
        buy_threshold=0.3,   # Lower threshold since heuristic generates -0.5% to 0.5%
        sell_threshold=0.3   # This will make signals more frequent
    )
    
    # 5. Generate signals with different market scenarios
    print("\n5. Generating signals with random predictions...")
    
    scenarios = [
        {
            'name': 'Market Scenario 1',
            'data': {
                'symbol': 'BTC-USD',
                'timestamp': '2025-01-07T16:00:00Z',
                'type': 'l2update',
                'changes': [['buy', '50000.0', '1.5'], ['sell', '50010.0', '2.0']],
                'best_bid': {'price': '50000.0', 'size': '1.5'},
                'best_ask': {'price': '50010.0', 'size': '2.0'},
                'spread': {'absolute': 10.0, 'percentage': 0.02, 'mid_price': '50005.0'},
                'top_bids': [['50000.0', '1.5'], ['49999.0', '1.0']],
                'top_asks': [['50010.0', '2.0'], ['50011.0', '1.5']]
            }
        },
        {
            'name': 'Market Scenario 2', 
            'data': {
                'symbol': 'ETH-USD',
                'timestamp': '2025-01-07T16:01:00Z',
                'type': 'l2update',
                'changes': [['buy', '3500.0', '5.0'], ['sell', '3502.0', '3.0']],
                'best_bid': {'price': '3500.0', 'size': '5.0'},
                'best_ask': {'price': '3502.0', 'size': '3.0'},
                'spread': {'absolute': 2.0, 'percentage': 0.057, 'mid_price': '3501.0'},
                'top_bids': [['3500.0', '5.0'], ['3499.0', '2.0']],
                'top_asks': [['3502.0', '3.0'], ['3503.0', '2.5']]
            }
        },
        {
            'name': 'Market Scenario 3',
            'data': {
                'symbol': 'SOL-USD',
                'timestamp': '2025-01-07T16:02:00Z',
                'type': 'l2update',
                'changes': [['buy', '200.0', '10.0'], ['sell', '200.5', '8.0']],
                'best_bid': {'price': '200.0', 'size': '10.0'},
                'best_ask': {'price': '200.5', 'size': '8.0'},
                'spread': {'absolute': 0.5, 'percentage': 0.25, 'mid_price': '200.25'},
                'top_bids': [['200.0', '10.0'], ['199.9', '5.0']],
                'top_asks': [['200.5', '8.0'], ['200.6', '6.0']]
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- {scenario['name']} ---")
        
        # Get model prediction
        prediction = model.predict(scenario['data'])
        print(f"🎲 Random prediction: {prediction:.4f}%")
        
        # Generate signal
        signal_generator.refresh(scenario['data'])
        
        # Show results
        print(f"📊 Market: {scenario['data']['symbol']}")
        print(f"💰 Price: ${float(scenario['data']['best_bid']['price']):.2f}")
        print(f"📈 Prediction: {prediction:+.4f}%")
        
        if abs(prediction) >= 0.3:
            action = "BUY" if prediction > 0 else "SELL"
            print(f"🚨 Signal: {action} (prediction exceeds ±0.3% threshold)")
        else:
            print(f"⏸️  Signal: HOLD (prediction within ±0.3% threshold)")
    
    # 6. Show batch predictions
    print(f"\n6. Batch predictions demonstration...")
    batch_predictions = model.generate_batch_predictions(10)
    print(f"🎲 10 random predictions: {[f'{p:.3f}%' for p in batch_predictions]}")
    print(f"📊 Range: {batch_predictions.min():.3f}% to {batch_predictions.max():.3f}%")
    print(f"📊 Mean: {batch_predictions.mean():.3f}%")
    
    # 7. Show trading results
    print(f"\n7. Trading Results Summary...")
    print(f"📋 Total trades executed: {len(trading_client.executed_trades)}")
    for i, trade in enumerate(trading_client.executed_trades):
        print(f"   Trade {i+1}: {trade['side']} {trade['size']} {trade['symbol']}")
    
    # 8. Model comparison
    print(f"\n8. Model Characteristics...")
    print(f"🎯 Model Type: {model.get_model_info()['model_type']}")
    print(f"🎲 Prediction Range: {model.get_model_info()['prediction_range']}")
    print(f"🔧 Random Seed: {model.get_model_info()['random_seed']}")
    print(f"📚 Training Data Size: {model.get_model_info()['training_data_size']}")
    
    print(f"\n✅ HeuristicModel demo completed!")
    print(f"💡 Key Points:")
    print(f"   • HeuristicModel generates random predictions between -0.5% and 0.5%")
    print(f"   • Useful for baseline testing and system validation")
    print(f"   • Follows same interface as other models (CatBoost, etc.)")
    print(f"   • Can be used to simulate market noise or random trading")

if __name__ == "__main__":
    demo_heuristic_model()
