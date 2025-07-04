#!/usr/bin/env python3
"""
Example integration showing how TradingClient works with OrderManager
This demonstrates the complete trading bot workflow.
"""

import sys
import time
from dotenv import load_dotenv

# Add src directory to path
sys.path.append('src')

from trading_client.trading_client import TradingClient

def example_trading_workflow():
    """
    Example workflow showing how TradingClient integrates with the trading bot architecture
    """
    print("🤖 Trading Bot Integration Example")
    print("=" * 40)
    
    try:
        # Step 1: Initialize TradingClient
        print("1. Initializing TradingClient...")
        client = TradingClient()
        print("✅ TradingClient ready")
        
        # Step 2: Check account status
        print("\n2. Checking account status...")
        balances = client.get_account_balance()
        
        usd_balance = balances['balances'].get('USD', {}).get('available', 0)
        eth_balance = balances['balances'].get('ETH', {}).get('available', 0)
        
        print(f"💰 USD Available: ${usd_balance}")
        print(f"🪙 ETH Available: {eth_balance} ETH")
        
        # Step 3: Get current ETH position
        print("\n3. Analyzing ETH-USD position...")
        position = client.view_position('ETH-USD')
        
        print(f"📊 Current ETH Price: ${position['current_price']}")
        print(f"📈 Position Value: ${position['position_value']:.2f}")
        print(f"🎯 ETH Holdings: {position['quantity_held']} ETH")
        
        # Step 4: Simulate trading signal processing
        print("\n4. Simulating trading signal processing...")
        
        # This is where your SignalGenerator would provide signals
        # For demo purposes, we'll simulate a simple signal
        mock_signal = {
            'action': 'buy',
            'product': 'ETH-USD',
            'confidence': 0.75,
            'suggested_amount': 0.001  # Very small for demo
        }
        
        print(f"📡 Received signal: {mock_signal['action'].upper()} {mock_signal['product']}")
        print(f"🎯 Confidence: {mock_signal['confidence']*100}%")
        print(f"💱 Suggested amount: {mock_signal['suggested_amount']} ETH")
        
        # Step 5: Risk management check
        print("\n5. Performing risk management checks...")
        
        # Check if we have enough balance
        if mock_signal['action'] == 'buy':
            required_usd = mock_signal['suggested_amount'] * position['current_price']
            if usd_balance < required_usd:
                print(f"❌ Insufficient USD balance. Need ${required_usd:.2f}, have ${usd_balance}")
                return
            else:
                print(f"✅ Sufficient balance for trade (${required_usd:.2f})")
        
        elif mock_signal['action'] == 'sell':
            if eth_balance < mock_signal['suggested_amount']:
                print(f"❌ Insufficient ETH balance. Need {mock_signal['suggested_amount']}, have {eth_balance}")
                return
            else:
                print(f"✅ Sufficient ETH for trade ({mock_signal['suggested_amount']} ETH)")
        
        # Step 6: Execute trade (commented out for safety)
        print("\n6. Trade execution (SIMULATION MODE)...")
        print("⚠️  Actual trading is disabled for safety")
        
        # Uncomment to execute real trades (BE VERY CAREFUL!)
        # if mock_signal['confidence'] > 0.7:  # Only trade on high confidence
        #     quantity = mock_signal['suggested_amount']
        #     if mock_signal['action'] == 'sell':
        #         quantity = -quantity  # Negative for sell
        #     
        #     trade_result = client.execute_trade(
        #         product=mock_signal['product'],
        #         quantity=quantity,
        #         order_type='market'
        #     )
        #     
        #     print(f"✅ Trade executed: {trade_result['order_id']}")
        #     
        #     # Monitor the trade
        #     time.sleep(2)  # Wait a moment
        #     status = client.get_trade_status(trade_result['order_id'])
        #     print(f"📊 Trade status: {status['status']}")
        #     print(f"📈 Filled: {status['filled_size']} / {status['size']}")
        
        print("🔄 Would execute market order for", mock_signal['suggested_amount'], "ETH")
        
        # Step 7: Portfolio update
        print("\n7. Portfolio status after trade simulation...")
        updated_portfolio = client.get_portfolio()
        
        print("📋 Current holdings:")
        for asset, details in updated_portfolio['holdings'].items():
            if details['quantity'] > 0:
                print(f"   {asset}: {details['quantity']:.6f}")
        
        print("\n🎉 Trading workflow completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in trading workflow: {str(e)}")
        return False
    
    return True

def demonstrate_order_manager_integration():
    """
    Show how TradingClient would integrate with OrderManager
    """
    print("\n" + "="*50)
    print("🔗 OrderManager Integration Example")
    print("="*50)
    
    print("""
    In your actual trading bot, the workflow would be:
    
    1. 📊 MarketDataFeed receives real-time price data
    2. 🧠 SignalGenerator processes data with ML models
    3. 📨 SignalGenerator publishes signals to OrderManager
    4. 🎯 OrderManager receives signals and converts to trades
    5. 💼 OrderManager uses TradingClient to execute trades
    6. 📈 TradingClient handles all Coinbase API interactions
    7. 🔄 Process repeats continuously
    
    Key Integration Points:
    
    OrderManager.convert_signals_to_trade() would:
    - Receive signals from SignalGenerator
    - Apply risk management rules
    - Determine trade size and type
    - Call TradingClient.execute_trade()
    
    OrderManager.manage_portfolio() would:
    - Use TradingClient.get_portfolio()
    - Use TradingClient.get_account_balance()
    - Monitor positions with TradingClient.view_position()
    
    OrderManager.direct_client_to_trade() would:
    - Execute trades via TradingClient.execute_trade()
    - Monitor status with TradingClient.get_trade_status()
    - Cancel if needed with TradingClient.cancel_trade()
    """)

if __name__ == "__main__":
    load_dotenv()
    
    print("Trading Bot - TradingClient Integration Demo")
    print("=" * 50)
    
    # Run the example workflow
    success = example_trading_workflow()
    
    # Show integration concepts
    demonstrate_order_manager_integration()
    
    print("\n" + "="*50)
    print("📝 Next Steps:")
    print("1. Set up your .env file with Coinbase sandbox credentials")
    print("2. Run: python test_trading_client.py")
    print("3. Implement your SignalGenerator logic")
    print("4. Update OrderManager to use TradingClient methods")
    print("5. Test thoroughly in sandbox before going live!")
    print("="*50)
    
    sys.exit(0 if success else 1)
