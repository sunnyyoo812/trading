#!/usr/bin/env python3
"""
Test script for TradingClient with Coinbase Advanced Trade API
Make sure to set up your .env file with sandbox credentials before running.
"""

import os
import sys
from dotenv import load_dotenv

# Add src directory to path
sys.path.append('src')

from trading_client.trading_client import TradingClient

def test_trading_client():
    """Test all TradingClient methods with sandbox environment"""
    
    print("🚀 Testing TradingClient with Coinbase Advanced Trade API (Sandbox)")
    print("=" * 60)
    
    try:
        # Initialize client
        print("1. Initializing TradingClient...")
        client = TradingClient()
        print("✅ TradingClient initialized successfully")
        
        # Test get_account_balance
        print("\n2. Testing get_account_balance()...")
        balances = client.get_account_balance()
        print(f"✅ Account balances retrieved: {len(balances['balances'])} currencies")
        for currency, details in balances['balances'].items():
            if details['balance'] > 0:
                print(f"   {currency}: {details['balance']} (available: {details['available']})")
        
        # Test get_portfolio
        print("\n3. Testing get_portfolio()...")
        portfolio = client.get_portfolio()
        print(f"✅ Portfolio retrieved: {portfolio['total_assets']} assets with balance")
        for asset, details in portfolio['holdings'].items():
            print(f"   {asset}: {details['quantity']} (available: {details['available']})")
        
        # Test view_position for ETH-USD
        print("\n4. Testing view_position('ETH-USD')...")
        position = client.view_position('ETH-USD')
        print(f"✅ ETH-USD position retrieved:")
        print(f"   Quantity held: {position['quantity_held']} ETH")
        print(f"   Current price: ${position['current_price']}")
        print(f"   Position value: ${position['position_value']:.2f}")
        print(f"   USD balance: ${position['quote_balance']}")
        
        # Test execute_trade (small test order)
        print("\n5. Testing execute_trade() - Small test market buy...")
        print("⚠️  This will place a real order in sandbox environment")
        
        # Uncomment the following lines to test actual trading (be careful!)
        # trade_result = client.execute_trade(
        #     product='ETH-USD',
        #     quantity=0.001,  # Very small amount for testing
        #     order_type='market'
        # )
        # print(f"✅ Trade executed: Order ID {trade_result['order_id']}")
        # 
        # # Test get_trade_status
        # print("\n6. Testing get_trade_status()...")
        # status = client.get_trade_status(trade_result['order_id'])
        # print(f"✅ Order status: {status['status']}")
        # print(f"   Filled: {status['filled_size']} / {status['size']}")
        
        print("\n🎉 All tests completed successfully!")
        print("\nNote: Trading tests are commented out for safety.")
        print("Uncomment the trading section in test_trading_client.py to test actual orders.")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you have created a .env file with your Coinbase sandbox credentials")
        print("2. Verify your API credentials are correct")
        print("3. Check that you're using sandbox environment")
        return False
    
    return True

def check_environment():
    """Check if environment is properly configured"""
    load_dotenv()
    
    required_vars = ['COINBASE_API_KEY', 'COINBASE_API_SECRET']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease create a .env file based on .env.example and add your credentials.")
        return False
    
    environment = os.getenv('COINBASE_ENVIRONMENT', 'sandbox')
    print(f"🔧 Environment: {environment}")
    
    if environment.lower() != 'sandbox':
        print("⚠️  WARNING: Not using sandbox environment!")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    
    return True

if __name__ == "__main__":
    print("Coinbase Advanced Trade API - TradingClient Test")
    print("=" * 50)
    
    if not check_environment():
        sys.exit(1)
    
    success = test_trading_client()
    sys.exit(0 if success else 1)
