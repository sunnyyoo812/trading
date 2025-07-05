#!/usr/bin/env python3
"""
Test and demonstration script for CatBoostTargetModel
Shows training, prediction, and evaluation capabilities
"""

import sys
import numpy as np
import pandas as pd
from typing import List, Dict
import logging

# Add src directory to path
sys.path.append('src')

try:
    from models.catboost_target_model import CatBoostTargetModel
    CATBOOST_AVAILABLE = True
except ImportError as e:
    print(f"❌ CatBoost not available: {e}")
    print("Please install required dependencies: pip install -r requirements.txt")
    CATBOOST_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_trading_data(n_samples: int = 1000) -> tuple:
    """
    Generate synthetic trading data for testing
    
    Parameters:
    - n_samples: Number of samples to generate
    
    Returns:
    - Tuple of (features_df, targets)
    """
    np.random.seed(42)
    
    # Generate base price data
    base_price = 2500
    price_changes = np.cumsum(np.random.normal(0, 1, n_samples))
    mid_prices = base_price + price_changes
    
    # Generate market features
    data = []
    targets = []
    
    for i in range(n_samples):
        mid_price = mid_prices[i]
        spread_pct = np.random.uniform(0.01, 0.1)  # 0.01% to 0.1%
        spread_abs = mid_price * spread_pct / 100
        
        bid_price = mid_price - spread_abs / 2
        ask_price = mid_price + spread_abs / 2
        
        # Generate volume data
        base_volume = np.random.uniform(5, 50)
        volume_imbalance = np.random.normal(0, 0.3)  # -0.3 to 0.3 typical range
        
        buy_volume = base_volume * (1 + volume_imbalance) / 2
        sell_volume = base_volume * (1 - volume_imbalance) / 2
        
        # Generate depth data
        bid_depth = np.random.uniform(10, 100)
        ask_depth = np.random.uniform(10, 100)
        depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        
        # Create feature dictionary
        features = {
            'bid_price': bid_price,
            'ask_price': ask_price,
            'mid_price': mid_price,
            'bid_size': np.random.uniform(1, 10),
            'ask_size': np.random.uniform(1, 10),
            'spread_absolute': spread_abs,
            'spread_percentage': spread_pct,
            'spread_mid_price': mid_price,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'total_volume': buy_volume + sell_volume,
            'volume_imbalance': volume_imbalance,
            'buy_orders': int(np.random.uniform(5, 20)),
            'sell_orders': int(np.random.uniform(5, 20)),
            'order_imbalance': np.random.normal(0, 0.2),
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'total_depth': bid_depth + ask_depth,
            'depth_imbalance': depth_imbalance,
            'weighted_mid_price': mid_price + np.random.normal(0, 0.1),
            'bid_levels': int(np.random.uniform(5, 15)),
            'ask_levels': int(np.random.uniform(5, 15))
        }
        
        # Generate target based on market conditions
        # Target represents expected price movement direction and strength
        momentum = volume_imbalance * 0.4 + depth_imbalance * 0.3
        volatility_factor = spread_pct / 0.05  # Normalize around 0.05%
        noise = np.random.normal(0, 0.1)
        
        target = momentum + volatility_factor * 0.1 + noise
        target = np.clip(target, -1.0, 1.0)  # Clip to reasonable range
        
        data.append(features)
        targets.append(target)
    
    return pd.DataFrame(data), np.array(targets)


def test_catboost_model():
    """Test CatBoost model functionality"""
    
    if not CATBOOST_AVAILABLE:
        return False
    
    print("🧪 Testing CatBoostTargetModel")
    print("=" * 60)
    
    try:
        # 1. Initialize model
        print("1. Initializing CatBoostTargetModel...")
        model = CatBoostTargetModel(
            iterations=100,  # Reduced for testing
            learning_rate=0.1,
            depth=4,
            verbose=False
        )
        print("✅ Model initialized successfully")
        
        # 2. Generate synthetic data
        print("\n2. Generating synthetic trading data...")
        features_df, targets = generate_synthetic_trading_data(500)
        print(f"✅ Generated {len(features_df)} samples with {len(features_df.columns)} features")
        
        # 3. Train model
        print("\n3. Training model...")
        model.train(features_df, targets)
        print("✅ Model training completed")
        
        # 4. Test prediction
        print("\n4. Testing prediction...")
        sample_features = features_df.iloc[0].to_dict()
        signals = model.predict(sample_features)
        print(f"✅ Generated {len(signals)} signals")
        print(f"   Sample signal: {signals[0]}")
        
        # 5. Evaluate model
        print("\n5. Evaluating model...")
        test_features, test_targets = generate_synthetic_trading_data(100)
        metrics = model.evaluate(test_features, test_targets)
        print("✅ Model evaluation completed")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   R² Score: {metrics['r2_score']:.4f}")
        
        # 6. Test feature importance
        print("\n6. Checking feature importance...")
        importance = model.get_feature_importance()
        if importance:
            print("✅ Top 5 most important features:")
            for i, (feature, score) in enumerate(list(importance.items())[:5]):
                print(f"   {i+1}. {feature}: {score:.4f}")
        
        # 7. Test model info
        print("\n7. Getting model information...")
        info = model.get_model_info()
        print("✅ Model info retrieved")
        print(f"   Model type: {info['model_type']}")
        print(f"   Feature count: {info['feature_count']}")
        print(f"   Is trained: {info['is_trained']}")
        
        # 8. Test save/load
        print("\n8. Testing model save/load...")
        model_path = "test_catboost_model.pkl"
        model.save_model(model_path)
        
        # Create new model and load
        new_model = CatBoostTargetModel()
        new_model.load_model(model_path)
        
        # Test prediction with loaded model
        loaded_signals = new_model.predict(sample_features)
        print("✅ Model save/load successful")
        print(f"   Original signal: {signals[0]['action']}")
        print(f"   Loaded signal: {loaded_signals[0]['action']}")
        
        # Cleanup
        import os
        if os.path.exists(model_path):
            os.remove(model_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_different_scenarios():
    """Demonstrate model behavior in different market scenarios"""
    
    if not CATBOOST_AVAILABLE:
        return
    
    print("\n🎯 CatBoost Model Scenario Demonstration")
    print("=" * 60)
    
    # Initialize and train model
    model = CatBoostTargetModel(iterations=200, verbose=False)
    features_df, targets = generate_synthetic_trading_data(1000)
    model.train(features_df, targets)
    
    # Test different market scenarios
    scenarios = [
        {
            'name': 'Strong Buy Pressure',
            'features': {
                'bid_price': 2500.0, 'ask_price': 2501.0, 'mid_price': 2500.5,
                'volume_imbalance': 0.6, 'depth_imbalance': 0.4,
                'spread_percentage': 0.04, 'total_volume': 50.0
            }
        },
        {
            'name': 'Strong Sell Pressure',
            'features': {
                'bid_price': 2500.0, 'ask_price': 2501.0, 'mid_price': 2500.5,
                'volume_imbalance': -0.6, 'depth_imbalance': -0.4,
                'spread_percentage': 0.04, 'total_volume': 50.0
            }
        },
        {
            'name': 'Tight Spread Market',
            'features': {
                'bid_price': 2500.4, 'ask_price': 2500.6, 'mid_price': 2500.5,
                'volume_imbalance': 0.1, 'depth_imbalance': 0.05,
                'spread_percentage': 0.008, 'total_volume': 30.0
            }
        },
        {
            'name': 'Wide Spread Market',
            'features': {
                'bid_price': 2499.0, 'ask_price': 2502.0, 'mid_price': 2500.5,
                'volume_imbalance': 0.0, 'depth_imbalance': 0.0,
                'spread_percentage': 0.12, 'total_volume': 20.0
            }
        },
        {
            'name': 'Balanced Market',
            'features': {
                'bid_price': 2500.0, 'ask_price': 2501.0, 'mid_price': 2500.5,
                'volume_imbalance': 0.0, 'depth_imbalance': 0.0,
                'spread_percentage': 0.04, 'total_volume': 40.0
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print("-" * 40)
        
        # Add default values for missing features
        features = scenario['features'].copy()
        default_features = {
            'bid_size': 2.0, 'ask_size': 2.0, 'spread_absolute': 1.0,
            'spread_mid_price': 2500.5, 'buy_volume': 25.0, 'sell_volume': 25.0,
            'buy_orders': 10, 'sell_orders': 10, 'order_imbalance': 0.0,
            'bid_depth': 50.0, 'ask_depth': 50.0, 'total_depth': 100.0,
            'weighted_mid_price': 2500.5, 'bid_levels': 10, 'ask_levels': 10
        }
        
        for key, value in default_features.items():
            if key not in features:
                features[key] = value
        
        # Get prediction
        signals = model.predict(features)
        signal = signals[0]
        
        print(f"Market Conditions:")
        print(f"  Volume Imbalance: {features['volume_imbalance']:.2f}")
        print(f"  Depth Imbalance: {features['depth_imbalance']:.2f}")
        print(f"  Spread: {features['spread_percentage']:.3f}%")
        
        print(f"Model Prediction:")
        print(f"  Action: {signal['action']}")
        print(f"  Confidence: {signal['confidence']:.3f}")
        print(f"  Prediction Score: {signal['prediction_score']:.3f}")
        print(f"  Reason: {signal['reason']}")


def performance_comparison():
    """Compare model performance with different parameters"""
    
    if not CATBOOST_AVAILABLE:
        return
    
    print("\n📈 Model Performance Comparison")
    print("=" * 60)
    
    # Generate test data
    train_features, train_targets = generate_synthetic_trading_data(800)
    test_features, test_targets = generate_synthetic_trading_data(200)
    
    # Test different model configurations
    configs = [
        {'name': 'Fast Model', 'iterations': 50, 'depth': 3, 'learning_rate': 0.2},
        {'name': 'Balanced Model', 'iterations': 200, 'depth': 6, 'learning_rate': 0.1},
        {'name': 'Deep Model', 'iterations': 300, 'depth': 8, 'learning_rate': 0.05}
    ]
    
    results = []
    
    for config in configs:
        print(f"\n🔧 Testing {config['name']}...")
        
        model = CatBoostTargetModel(
            iterations=config['iterations'],
            depth=config['depth'],
            learning_rate=config['learning_rate'],
            verbose=False
        )
        
        # Train model
        model.train(train_features, train_targets)
        
        # Evaluate
        metrics = model.evaluate(test_features, test_targets)
        
        result = {
            'name': config['name'],
            'rmse': metrics['rmse'],
            'r2_score': metrics['r2_score'],
            'mae': metrics['mae']
        }
        results.append(result)
        
        print(f"  RMSE: {result['rmse']:.4f}")
        print(f"  R² Score: {result['r2_score']:.4f}")
        print(f"  MAE: {result['mae']:.4f}")
    
    # Find best model
    best_model = min(results, key=lambda x: x['rmse'])
    print(f"\n🏆 Best Model: {best_model['name']}")
    print(f"   RMSE: {best_model['rmse']:.4f}")
    print(f"   R² Score: {best_model['r2_score']:.4f}")


if __name__ == '__main__':
    print("CatBoost Target Model Test Suite")
    print("=" * 60)
    
    if not CATBOOST_AVAILABLE:
        print("❌ CatBoost dependencies not available")
        print("Please install: pip install -r requirements.txt")
        exit(1)
    
    try:
        # Run basic functionality test
        success = test_catboost_model()
        
        if success:
            # Run scenario demonstrations
            demo_different_scenarios()
            
            # Run performance comparison
            performance_comparison()
            
            print("\n🎉 All tests completed successfully!")
            print("\n📝 Key Features Demonstrated:")
            print("  • CatBoost regressor training and prediction")
            print("  • Feature engineering and preparation")
            print("  • Signal generation from predictions")
            print("  • Model evaluation and metrics")
            print("  • Feature importance analysis")
            print("  • Model save/load functionality")
            print("  • Different market scenario handling")
            print("  • Performance comparison across configurations")
            
        else:
            print("❌ Basic tests failed")
            exit(1)
            
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    exit(0)
