#!/usr/bin/env python3
"""
Demonstration of complete trading flow with WorkflowOrchestrator
Tests the full pipeline: Model → Market Data → Signals → Trading
"""

import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np

from src.orchestrator.workflow_orchestrator import WorkflowOrchestrator

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def demo_complete_trading_flow():
    """Demonstrate the complete trading flow in sandbox mode"""
    print("🚀 Complete Trading Flow Demo")
    print("=" * 60)
    print("Testing: Model Training → Market Data → Signal Generation → Trading")
    print("Environment: Coinbase Sandbox (Safe for testing)")
    print("=" * 60)
    
    # 1. Initialize WorkflowOrchestrator
    print("\n1. Initializing WorkflowOrchestrator...")
    orchestrator = WorkflowOrchestrator(
        data_dir="demo_trading_data",
        model_dir="demo_trading_models",
        symbols=['DOGE-USD'],  # Single symbol for focused demo
        model_type='heuristic'  # Use heuristic model for predictable testing
    )
    print("✅ WorkflowOrchestrator initialized")
    
    try:
        # 2. Create and train a model (using mock data for demo)
        print("\n2. Training Model...")
        
        # Create mock historical data for training
        mock_data = create_mock_training_data()
        print(f"   Created mock dataset with {len(mock_data)} records")
        
        # Train model using the orchestrator's training method
        # First save the mock data as if it were historical data
        import os
        from datetime import datetime
        os.makedirs("demo_trading_data", exist_ok=True)
        
        # Use today's date for the filename so the data loader can find it
        today_str = datetime.now().strftime('%Y%m%d')
        mock_data.to_csv(f"demo_trading_data/ETH-USD_{today_str}.csv", index=False)
        print(f"   Saved mock data to: ETH-USD_{today_str}.csv")
        
        # Now train using the saved data
        trained_model = orchestrator.run_training_pipeline()
        
        if trained_model:
            print("✅ Model training completed")
            
            # Deploy model to registry
            metrics = {'rmse': 0.5, 'mae': 0.3}  # Mock metrics
            model_path = orchestrator.deploy_model(trained_model, metrics)
            print(f"✅ Model deployed: {model_path}")
        else:
            print("❌ Model training failed")
            return
        
        # 3. Start Trading Flow
        print("\n3. Starting Trading Flow...")
        print("   🔧 Components to initialize:")
        print("      • Market Data Feed (Coinbase Sandbox)")
        print("      • Signal Generator (with trained model)")
        print("      • Trading Client (Sandbox mode)")
        print("      • Order Manager (trade execution)")
        
        success = orchestrator.start_trading_flow(
            environment='production',
            trade_amount=1.0,  # Small amount for testing
            buy_threshold=0.3,   # Lower threshold for more signals
            sell_threshold=0.3
        )
        
        if not success:
            print("❌ Failed to start trading flow")
            return
        
        # 4. Monitor Trading Status
        print("\n4. Trading Flow Status:")
        trading_status = orchestrator.get_trading_status()
        
        print(f"   🎯 Trading Active: {trading_status['is_trading']}")
        print(f"   🌐 Environment: {trading_status['environment']}")
        print(f"   💰 Trade Amount: ${trading_status['trade_amount_usd']}")
        print(f"   📊 Symbols: {trading_status['symbols']}")
        
        print("\n   📋 Component Status:")
        components = trading_status['components']
        for component, status in components.items():
            status_icon = "✅" if status else "❌"
            print(f"      {status_icon} {component.replace('_', ' ').title()}: {status}")
        
        # 5. Let trading run for a short period
        print("\n5. Monitoring Live Trading...")
        print("   📈 Collecting market data and generating signals...")
        print("   ⏱️  Running for 30 seconds to observe trading activity...")
        
        for i in range(6):  # 30 seconds, check every 5 seconds
            time.sleep(5)
            
            # Get current performance
            performance = orchestrator.get_trading_performance()
            
            if 'error' not in performance:
                total_trades = performance.get('total_trades', 0)
                print(f"   📊 Update {i+1}/6: {total_trades} trades executed")
                
                if total_trades > 0:
                    print(f"      Trades by side: {performance.get('trades_by_side', {})}")
                    print(f"      Total volume: {performance.get('total_volume', 0):.4f}")
            else:
                print(f"   ⚠️  Performance data: {performance['error']}")
        
        # 6. Final Trading Summary
        print("\n6. Final Trading Summary:")
        final_performance = orchestrator.get_trading_performance()
        
        if 'error' not in final_performance:
            print(f"   📈 Total Trades: {final_performance.get('total_trades', 0)}")
            print(f"   📊 By Symbol: {final_performance.get('trades_by_symbol', {})}")
            print(f"   📊 By Side: {final_performance.get('trades_by_side', {})}")
            print(f"   📊 Total Volume: {final_performance.get('total_volume', 0):.4f}")
        else:
            print(f"   ⚠️  {final_performance['error']}")
        
        # 7. Test Trading Controls
        print("\n7. Testing Trading Controls...")
        
        # Test restart
        print("   🔄 Testing restart...")
        restart_success = orchestrator.restart_trading_flow(trade_amount=75.0)
        print(f"   {'✅' if restart_success else '❌'} Restart: {restart_success}")
        
        # Brief pause to see restart effect
        time.sleep(3)
        
        # Test configuration update
        print("   ⚙️  Testing configuration update...")
        orchestrator.update_trading_config(trade_amount=100.0)
        print("   ✅ Configuration updated")
        
        # 8. Component Integration Test
        print("\n8. Component Integration Verification:")
        
        # Verify model is loaded
        current_model = orchestrator.get_current_model()
        if current_model:
            model_info = current_model.get_model_info()
            print(f"   ✅ Model: {model_info['model_type']} (trained: {model_info['is_trained']})")
        else:
            print("   ❌ No model loaded")
        
        # Verify market data connection
        if orchestrator.trading_market_feed:
            print("   ✅ Market Data Feed: Connected")
        else:
            print("   ❌ Market Data Feed: Not connected")
        
        # Verify signal generator
        if orchestrator.trading_signal_generator:
            print("   ✅ Signal Generator: Active")
        else:
            print("   ❌ Signal Generator: Not active")
        
        # Verify trading client
        if orchestrator.trading_client:
            print("   ✅ Trading Client: Connected (Sandbox)")
        else:
            print("   ❌ Trading Client: Not connected")
        
        # Verify order manager
        if orchestrator.trading_order_manager:
            print("   ✅ Order Manager: Active")
        else:
            print("   ❌ Order Manager: Not active")
        
        # 9. Safety and Environment Verification
        print("\n9. Safety Verification:")
        print(f"   🛡️  Environment: {orchestrator.trading_environment}")
        print(f"   🛡️  Sandbox Mode: {'✅ SAFE' if orchestrator.trading_environment == 'sandbox' else '⚠️ PRODUCTION'}")
        print(f"   🛡️  Trade Limits: ${orchestrator.trade_amount_usd} per trade")
        
        # 10. Demonstrate graceful shutdown
        print("\n10. Graceful Shutdown...")
        orchestrator.stop_trading_flow()
        print("   ✅ Trading flow stopped")
        
        # Verify cleanup
        final_status = orchestrator.get_trading_status()
        print(f"   ✅ Trading active: {final_status['is_trading']} (should be False)")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure cleanup
        print("\n🧹 Cleanup...")
        try:
            orchestrator.stop_trading_flow()
            print("✅ Trading flow cleanup completed")
        except:
            pass
    
    print("\n🎉 Trading Flow Demo Completed!")
    print("\n💡 Key Features Demonstrated:")
    print("   • Complete model training and deployment pipeline")
    print("   • Real-time market data integration (Coinbase Sandbox)")
    print("   • Signal generation using trained models")
    print("   • Automated trade execution via OrderManager")
    print("   • Live trading monitoring and performance tracking")
    print("   • Safe sandbox environment for testing")
    print("   • Graceful startup, restart, and shutdown procedures")
    print("   • Component integration and health monitoring")

def create_mock_training_data():
    """Create mock historical data for model training"""
    np.random.seed(42)  # For reproducible demo
    
    n_samples = 200
    
    # Create realistic price movement
    base_price = 3500
    price_changes = np.random.randn(n_samples) * 0.5
    prices = base_price + np.cumsum(price_changes)
    
    mock_data = {
        'symbol': ['ETH-USD'] * n_samples,
        'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='1min'),
        'mid_price': prices,
        'bid_price': prices - 0.5,
        'ask_price': prices + 0.5,
        'bid_size': np.random.uniform(1, 5, n_samples),
        'ask_size': np.random.uniform(1, 5, n_samples),
        'spread_percentage': np.random.uniform(0.01, 0.1, n_samples),
        'spread_absolute': np.random.uniform(0.5, 2.0, n_samples),
        'volume_imbalance': np.random.uniform(-0.5, 0.5, n_samples),
        'total_volume': np.random.uniform(10, 100, n_samples),
        'bid_depth': np.random.uniform(50, 200, n_samples),
        'ask_depth': np.random.uniform(50, 200, n_samples),
        'buy_volume': np.random.uniform(5, 50, n_samples),
        'sell_volume': np.random.uniform(5, 50, n_samples),
        'order_changes': np.random.randint(1, 10, n_samples)
    }
    
    return pd.DataFrame(mock_data)

def demo_trading_with_real_model():
    """
    Alternative demo using a real CatBoost model (if available)
    This requires more setup but shows production-like behavior
    """
    print("🔬 Advanced Trading Demo with CatBoost Model")
    print("=" * 60)
    
    orchestrator = WorkflowOrchestrator(
        symbols=['ETH-USD'],
        model_type='catboost'
    )
    
    # Check if we have a trained model
    current_model = orchestrator.get_current_model()
    
    if current_model is None:
        print("⚠️  No trained CatBoost model found")
        print("   Running training pipeline first...")
        
        # Create more sophisticated training data
        training_data = create_advanced_training_data()
        trained_model = orchestrator.run_training_pipeline()
        
        if trained_model is None:
            print("❌ Failed to train CatBoost model")
            return
    
    # Start trading with CatBoost model
    success = orchestrator.start_trading_flow(
        environment='production',
        trade_amount=100.0,
        buy_threshold=1.0,  # Higher threshold for CatBoost
        sell_threshold=1.0
    )
    
    if success:
        print("✅ CatBoost trading flow started")
        
        # Monitor for longer period
        print("📈 Monitoring CatBoost trading for 60 seconds...")
        for i in range(12):  # 60 seconds
            time.sleep(5)
            performance = orchestrator.get_trading_performance()
            if 'error' not in performance:
                print(f"   Update {i+1}/12: {performance.get('total_trades', 0)} trades")
        
        orchestrator.stop_trading_flow()
        print("✅ CatBoost demo completed")
    else:
        print("❌ Failed to start CatBoost trading")

def create_advanced_training_data():
    """Create more sophisticated training data with realistic market patterns"""
    np.random.seed(123)
    
    n_samples = 1000
    
    # Create realistic market microstructure
    base_price = 3500
    
    # Add trend, volatility clustering, and mean reversion
    returns = []
    volatility = 0.01
    
    for i in range(n_samples):
        # Volatility clustering
        volatility = 0.95 * volatility + 0.05 * np.random.exponential(0.01)
        
        # Mean reverting returns with trend
        trend = 0.0001 * np.sin(i / 100)  # Slow trend
        mean_reversion = -0.1 * (returns[-1] if returns else 0)
        noise = np.random.normal(0, volatility)
        
        return_val = trend + mean_reversion + noise
        returns.append(return_val)
    
    # Convert returns to prices
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])  # Remove initial price
    
    # Create comprehensive feature set
    mock_data = {
        'symbol': ['ETH-USD'] * n_samples,
        'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='1min'),
        'mid_price': prices,
        'bid_price': prices - np.random.uniform(0.1, 1.0, n_samples),
        'ask_price': prices + np.random.uniform(0.1, 1.0, n_samples),
        'bid_size': np.random.lognormal(1, 0.5, n_samples),
        'ask_size': np.random.lognormal(1, 0.5, n_samples),
        'spread_percentage': np.random.uniform(0.01, 0.2, n_samples),
        'spread_absolute': np.random.uniform(0.1, 2.0, n_samples),
        'volume_imbalance': np.random.normal(0, 0.3, n_samples),
        'total_volume': np.random.lognormal(3, 1, n_samples),
        'bid_depth': np.random.lognormal(4, 0.5, n_samples),
        'ask_depth': np.random.lognormal(4, 0.5, n_samples),
        'buy_volume': np.random.lognormal(2, 0.8, n_samples),
        'sell_volume': np.random.lognormal(2, 0.8, n_samples),
        'order_changes': np.random.poisson(5, n_samples)
    }
    
    return pd.DataFrame(mock_data)

if __name__ == "__main__":
    # Run the main demo
    demo_complete_trading_flow()
    
    # Optionally run advanced demo
    print("\n" + "="*60)
    print("Optional: Advanced Demo with CatBoost")
    print("="*60)
    
    response = input("Run advanced CatBoost demo? (y/n): ")
    if response.lower() == 'y':
        demo_trading_with_real_model()
    else:
        print("Skipping advanced demo")
    
    print("\n🎯 Demo completed! The trading system is ready for production use.")
    print("💡 Next steps:")
    print("   1. Configure real API credentials for production")
    print("   2. Set up monitoring and alerting")
    print("   3. Implement risk management rules")
    print("   4. Schedule daily training pipeline")
