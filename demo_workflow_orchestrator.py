#!/usr/bin/env python3
"""
Demonstration of WorkflowOrchestrator with scheduled training pipeline
"""

import time
import logging
from datetime import datetime, timedelta
import pytz

from src.orchestrator.workflow_orchestrator import WorkflowOrchestrator

# Configure logging to see detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def demo_workflow_orchestrator():
    """Demonstrate the WorkflowOrchestrator functionality"""
    print("🚀 WorkflowOrchestrator Demo")
    print("=" * 60)
    
    # 1. Initialize WorkflowOrchestrator
    print("\n1. Initializing WorkflowOrchestrator...")
    orchestrator = WorkflowOrchestrator(
        data_dir="demo_data/historical",
        model_dir="demo_models",
        symbols=['ETH-USD'],  # Single symbol for demo
        model_type='catboost'
    )
    
    print("✅ WorkflowOrchestrator initialized")
    
    # 2. Show initial status
    print("\n2. Initial Status:")
    status = orchestrator.get_status()
    print(f"   Running: {status['is_running']}")
    print(f"   Symbols: {status['symbols']}")
    print(f"   Model Type: {status['model_type']}")
    print(f"   Registered Models: {status['registered_models']}")
    
    # 3. Start data collection (without scheduler for demo)
    print("\n3. Starting data collection...")
    orchestrator.data_generator.start_data_collection(orchestrator.symbols)
    
    # Wait a moment for some data to be collected
    print("   Collecting market data for 10 seconds...")
    time.sleep(10)
    
    # 4. Check data collection status
    print("\n4. Data Collection Status:")
    data_status = orchestrator.data_generator.get_collection_status()
    print(f"   Is Collecting: {data_status['is_collecting']}")
    print(f"   Symbols: {data_status['symbols']}")
    print(f"   Buffer Sizes: {data_status['buffer_sizes']}")
    
    # 5. Manual data save (simulating 4 PM job)
    print("\n5. Manual Data Save (simulating 4 PM scheduled job)...")
    saved_files = orchestrator.manual_data_save()
    
    if saved_files:
        print("✅ Data saved successfully:")
        for symbol, filepath in saved_files.items():
            print(f"   {symbol}: {filepath}")
    else:
        print("⚠️ No data was saved (may need more collection time)")
    
    # 6. Manual training (simulating 5 PM job)
    print("\n6. Manual Training (simulating 5 PM scheduled job)...")
    
    # First check if we have historical data
    df = orchestrator.load_historical_data()
    
    if df.empty:
        print("   No historical data found, creating mock data for demo...")
        # Create some mock historical data for demonstration
        import pandas as pd
        import numpy as np
        
        mock_data = {
            'symbol': ['ETH-USD'] * 200,
            'timestamp': pd.date_range('2025-01-01', periods=200, freq='1min'),
            'mid_price': 3500 + np.cumsum(np.random.randn(200) * 0.1),
            'bid_price': lambda x: x - 0.5,
            'ask_price': lambda x: x + 0.5,
            'bid_size': np.random.uniform(1, 5, 200),
            'ask_size': np.random.uniform(1, 5, 200),
            'spread_percentage': np.random.uniform(0.01, 0.1, 200),
            'volume_imbalance': np.random.uniform(-0.5, 0.5, 200),
            'total_volume': np.random.uniform(10, 100, 200),
            'bid_depth': np.random.uniform(50, 200, 200),
            'ask_depth': np.random.uniform(50, 200, 200)
        }
        
        df = pd.DataFrame(mock_data)
        df['bid_price'] = df['mid_price'] - 0.5
        df['ask_price'] = df['mid_price'] + 0.5
        
        print(f"   Created mock dataset with {len(df)} records")
    
    # Run training pipeline
    trained_model = orchestrator.run_training_pipeline()
    
    if trained_model:
        print("✅ Training completed successfully!")
        
        # Show model info
        if hasattr(trained_model, 'get_model_info'):
            model_info = trained_model.get_model_info()
            print(f"   Model Type: {model_info.get('model_type', 'Unknown')}")
            print(f"   Is Trained: {model_info.get('is_trained', False)}")
    else:
        print("❌ Training failed")
    
    # 7. Show final status
    print("\n7. Final Status:")
    final_status = orchestrator.get_status()
    print(f"   Last Data Save: {final_status['last_data_save']}")
    print(f"   Last Training: {final_status['last_training']}")
    print(f"   Registered Models: {final_status['registered_models']}")
    
    # 8. Test getting current model
    print("\n8. Testing Model Retrieval:")
    current_model = orchestrator.get_current_model()
    if current_model:
        print("✅ Current model retrieved successfully")
        
        # Test a prediction
        test_data = {
            'mid_price': 3500.0,
            'bid_price': 3499.5,
            'ask_price': 3500.5,
            'spread_percentage': 0.03,
            'volume_imbalance': 0.1
        }
        
        prediction = current_model.predict(test_data)
        print(f"   Test prediction: {prediction:.4f}% price change")
    else:
        print("❌ No current model available")
    
    # 9. Demonstrate scheduler (without actually starting it)
    print("\n9. Scheduler Configuration:")
    print("   If scheduler were started:")
    print("   📅 Data Save: Daily at 4:00 PM EST")
    print("   🤖 Model Training: Daily at 5:00 PM EST")
    print("   📊 Data Collection: Continuous during market hours")
    
    # Show what the next scheduled times would be
    est_tz = pytz.timezone('US/Eastern')
    now = datetime.now(est_tz)
    
    # Calculate next 4 PM
    next_4pm = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if next_4pm <= now:
        next_4pm += timedelta(days=1)
    
    # Calculate next 5 PM
    next_5pm = now.replace(hour=17, minute=0, second=0, microsecond=0)
    if next_5pm <= now:
        next_5pm += timedelta(days=1)
    
    print(f"   Next Data Save: {next_4pm.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"   Next Training: {next_5pm.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # 10. Cleanup
    print("\n10. Cleanup...")
    orchestrator.data_generator.stop_data_collection()
    print("✅ Data collection stopped")
    
    print("\n🎉 WorkflowOrchestrator demo completed!")
    print("\n💡 Key Features Demonstrated:")
    print("   • Automated data collection from live market feeds")
    print("   • Scheduled data saving (4 PM EST daily)")
    print("   • Scheduled model training (5 PM EST daily)")
    print("   • Model registry management")
    print("   • Error handling and status monitoring")
    print("   • Manual trigger capabilities for testing")

def demo_scheduler_start():
    """
    Demonstrate starting the actual scheduler (use with caution)
    This will start real scheduled jobs
    """
    print("⚠️ Starting REAL scheduler with actual scheduled jobs...")
    print("This will run data collection and training at scheduled times.")
    
    response = input("Are you sure you want to start the scheduler? (yes/no): ")
    if response.lower() != 'yes':
        print("Scheduler start cancelled")
        return
    
    orchestrator = WorkflowOrchestrator()
    
    try:
        # Start the scheduler
        orchestrator.start_scheduler()
        
        print("✅ Scheduler started successfully!")
        print("📊 Data collection is now running...")
        print("⏰ Scheduled jobs:")
        
        status = orchestrator.get_status()
        for job in status['scheduled_jobs']:
            print(f"   {job['name']}: {job['next_run']}")
        
        print("\nPress Ctrl+C to stop the scheduler...")
        
        # Keep running until interrupted
        while True:
            time.sleep(60)  # Check every minute
            status = orchestrator.get_status()
            print(f"Status check - Data buffers: {status['data_collection']['buffer_sizes']}")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping scheduler...")
        orchestrator.stop_scheduler()
        print("✅ Scheduler stopped")

if __name__ == "__main__":
    # Run the main demo
    demo_workflow_orchestrator()
    
    # Optionally start the real scheduler
    print("\n" + "="*60)
    print("Optional: Start Real Scheduler")
    print("="*60)
    demo_scheduler_start()
