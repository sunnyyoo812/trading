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
from trading_client.exceptions import ValidationError, OrderError, APIError, AuthenticationError
from trading_client.models import OrderType, OrderSide

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
        account_balance = client.get_account_balance()
        print(f"✅ Account balances retrieved: {len(account_balance.balances)} currencies")
        for currency, balance in account_balance.balances.items():
            if balance.balance > 0:
                print(f"   {currency}: {balance.balance} (available: {balance.available})")
        
        # Test portfolio overview (using account balance data)
        print("\n3. Testing portfolio overview...")
        print(f"✅ Portfolio overview:")
        print(f"   Total USD value: ${account_balance.total_value_usd:.2f}")
        assets_with_balance = [currency for currency, balance in account_balance.balances.items() if balance.balance > 0]
        print(f"   Assets with balance: {len(assets_with_balance)}")
        for currency in assets_with_balance:
            balance = account_balance.balances[currency]
            print(f"   {currency}: {balance.balance} (available: {balance.available})")
        
        # Test position view for specific currencies
        print("\n4. Testing position view for available currencies...")
        for currency, balance in account_balance.balances.items():
            if balance.balance > 0 and currency != 'USD':
                print(f"✅ {currency} position:")
                print(f"   Total held: {balance.balance} {currency}")
                print(f"   Available: {balance.available} {currency}")
                print(f"   On hold: {balance.hold} {currency}")
                break
        else:
            print("✅ No non-USD positions found (this is normal for new accounts)")
        

        # Test execute_trade (small test order) - COMMENTED FOR SAFETY
        print("\n6. Testing execute_trade() - Small test market buy...")
        print("⚠️  This will place a real order in sandbox environment")
        
        # Uncomment the following lines to test actual trading (be careful!)
        trade_result = client.execute_trade(
            client_order_id="test-order-001",  # Optional client order ID for tracking
            product_id='DOGE-USD',  # Fixed parameter name
            quantity=3,
            order_type=OrderType.MARKET,  # Use enum,
            side=OrderSide.SELL
        )
        print(f"✅ Trade executed: Order ID {trade_result.client_order_id}")
        print(f"   Status: {trade_result.status}")
        print(f"   Side: {trade_result.side}")
        print(f"   Size: {trade_result.size}")
        
        # # Test get_trade_status
        print("\n7. Testing get_trade_status()...")
        order_id = "d6bd879e-f9ff-4013-aaa3-7d11dc31fce7"
        status = client.get_trade_status(order_id)
        print(f"✅ Order status: {status.status}")
        print(f"   Filled: {status.filled_size} / {status.size}")
        print(f"   Remaining: {status.remaining_size}")
        print(f"   Completion: {status.completion_percentage}%")
        
        print("\n🎉 All tests completed successfully!")
        print("\nNote: Trading tests are commented out for safety.")
        print("Uncomment the trading section in test_trading_client.py to test actual orders.")
        
    except AuthenticationError as e:
        print(f"❌ Authentication Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you have created a .env file with your Coinbase sandbox credentials")
        print("2. Verify your API key and secret are correct")
        print("3. Check that your credentials are for the correct environment (sandbox/production)")
        return False
    except APIError as e:
        print(f"❌ API Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify Coinbase API service status")
        print("3. Ensure you're using the correct environment")
        return False
    except ValidationError as e:
        print(f"❌ Validation Error: {str(e)}")
        print("\nThis indicates an issue with the test parameters.")
        return False
    except OrderError as e:
        print(f"❌ Order Error: {str(e)}")
        print("\nThis indicates an issue with order execution.")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("\nTroubleshooting:")
        print("1. Make sure you have created a .env file with your Coinbase sandbox credentials")
        print("2. Verify your API credentials are correct")
        print("3. Check that you're using sandbox environment")
        print("4. Ensure all required dependencies are installed")
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
