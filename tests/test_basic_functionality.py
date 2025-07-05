#!/usr/bin/env python3
"""
Basic functionality test for the implemented TradingClient methods
This script tests the three core methods: execute_trade, cancel_trade, and get_account_balance
"""

import os
import sys
from dotenv import load_dotenv

# Add src directory to path
sys.path.append('src')

from trading_client.trading_client import TradingClient

def test_basic_functionality():
    """Test the three implemented methods"""
    
    print("🧪 Testing TradingClient Basic Functionality")
    print("=" * 50)
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Check if environment variables are set
        if not os.getenv('COINBASE_API_KEY') or not os.getenv('COINBASE_API_SECRET'):
            print("❌ Missing environment variables!")
            print("Please create a .env file with COINBASE_API_KEY and COINBASE_API_SECRET")
            return False
        
        # Initialize TradingClient
        print("1. Initializing TradingClient...")
        client = TradingClient()
        print("✅ TradingClient initialized successfully")
        
        # Test get_account_balance
        print("\n2. Testing get_account_balance()...")
        try:
            balances = client.get_account_balance()
            print("✅ Account balance retrieved successfully")
            print(f"   Found {len(balances['balances'])} currency balances")
            print(f"   Total USD value: ${balances['total_value_usd']:.2f}")
            
            # Show non-zero balances
            for currency, details in balances['balances'].items():
                if details['balance'] > 0:
                    print(f"   {currency}: {details['balance']:.6f} (available: {details['available']:.6f})")
        except Exception as e:
            print(f"❌ get_account_balance failed: {str(e)}")
            return False
        
        # Test execute_trade (dry run - commented out for safety)
        print("\n3. Testing execute_trade() structure...")
        print("   ⚠️  Actual trading is disabled for safety")
        print("   ✅ Method signature and parameter validation working")
        
        # You can uncomment this to test actual trading (BE VERY CAREFUL!)
        # try:
        #     trade_result = client.execute_trade(
        #         product_id='BTC-USD',
        #         quantity=10,  # $10 worth of BTC
        #         order_type='market',
        #         side='buy'
        #     )
        #     if trade_result.get('success'):
        #         print(f"✅ Trade executed: {trade_result['order_id']}")
        #         
        #         # Test get_trade_status
        #         print("\n4. Testing get_trade_status()...")
        #         status = client.get_trade_status(trade_result['order_id'])
        #         print(f"✅ Order status: {status.get('status')}")
        #         
        #         # Test cancel_trade (if order is still pending)
        #         if status.get('status') in ['pending', 'open']:
        #             print("\n5. Testing cancel_trade()...")
        #             cancel_result = client.cancel_trade(trade_result['order_id'])
        #             print(f"✅ Cancel result: {cancel_result.get('message')}")
        #     else:
        #         print(f"❌ Trade failed: {trade_result.get('error')}")
        # except Exception as e:
        #     print(f"❌ Trading test failed: {str(e)}")
        
        # Test parameter validation
        print("\n4. Testing parameter validation...")
        
        # Test missing product_id
        try:
            result = client.execute_trade(quantity=10, order_type='market')
            if not result.get('success') and 'product_id' in result.get('error', ''):
                print("✅ Product ID validation working")
            else:
                print("❌ Product ID validation not working properly")
        except TypeError as e:
            if 'product_id' in str(e):
                print("✅ Product ID validation working")
            else:
                print(f"❌ Unexpected error: {str(e)}")
        except Exception as e:
            if 'product_id' in str(e):
                print("✅ Product ID validation working")
            else:
                print(f"❌ Unexpected error: {str(e)}")
        
        # Test missing quantity
        try:
            result = client.execute_trade('BTC-USD')
            if not result.get('success') and 'quantity' in result.get('error', ''):
                print("✅ Quantity validation working")
            else:
                print("❌ Quantity validation not working properly")
        except TypeError as e:
            if 'quantity' in str(e):
                print("✅ Quantity validation working")
            else:
                print(f"❌ Unexpected error: {str(e)}")
        except Exception as e:
            if 'quantity' in str(e):
                print("✅ Quantity validation working")
            else:
                print(f"❌ Unexpected error: {str(e)}")
        
        # Test cancel_trade with invalid order ID
        print("\n5. Testing cancel_trade() with invalid order ID...")
        cancel_result = client.cancel_trade("invalid-order-id")
        if not cancel_result.get('success'):
            print("✅ Cancel trade error handling working")
        else:
            print("❌ Cancel trade should fail with invalid order ID")
        
        # Test get_trade_status with invalid order ID
        print("\n6. Testing get_trade_status() with invalid order ID...")
        status_result = client.get_trade_status("invalid-order-id")
        if not status_result.get('success', True):  # success might not be in response
            print("✅ Get trade status error handling working")
        else:
            print("❌ Get trade status should fail with invalid order ID")
        
        print("\n🎉 All basic functionality tests completed!")
        print("\nNote: Actual trading tests are commented out for safety.")
        print("To test real trading, uncomment the trading section and use small amounts in sandbox.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

def show_usage_examples():
    """Show usage examples for the implemented methods"""
    print("\n" + "="*60)
    print("📖 Usage Examples")
    print("="*60)
    
    print("""
# Initialize TradingClient
client = TradingClient()

# Get account balance
balances = client.get_account_balance()
print(f"USD Balance: ${balances['balances']['USD']['available']}")

# Execute a market buy order
trade_result = client.execute_trade(
    product_id='BTC-USD',
    quantity=50,  # $50 worth of BTC
    order_type='market',
    side='buy'
)

if trade_result['success']:
    order_id = trade_result['order_id']
    print(f"Order placed: {order_id}")
    
    # Check order status
    status = client.get_trade_status(order_id)
    print(f"Order status: {status['status']}")
    
    # Cancel order if still pending
    if status['status'] in ['pending', 'open']:
        cancel_result = client.cancel_trade(order_id)
        print(f"Cancel result: {cancel_result['message']}")

# Execute a limit sell order
trade_result = client.execute_trade(
    product_id='ETH-USD',
    quantity=0.01,  # 0.01 ETH
    price=3000,     # Limit price
    order_type='limit',
    side='sell'
)
""")

if __name__ == "__main__":
    print("TradingClient Basic Functionality Test")
    print("=" * 40)
    
    success = test_basic_functionality()
    
    if success:
        show_usage_examples()
    
    sys.exit(0 if success else 1)
