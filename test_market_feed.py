#!/usr/bin/env python3
"""
Test script for the new Coinbase WebSocket MarketDataFeed implementation
This demonstrates Level 2 order book data streaming for ETH-USD
"""

import sys
import time
import json
from dotenv import load_dotenv

# Add src directory to path
sys.path.append('src')

from market_feed.market_feed import MarketDataFeed

def market_data_callback(data):
    """
    Callback function to handle market data updates
    This simulates how SignalGenerator would receive the data
    """
    print("\n" + "="*60)
    print(f"📊 Market Data Update: {data['symbol']}")
    print(f"🕐 Timestamp: {data['timestamp']}")
    print(f"📈 Type: {data['type']}")
    
    # Show best bid/ask
    if data['best_bid'] and data['best_ask']:
        print(f"💰 Best Bid: ${data['best_bid']['price']} (Size: {data['best_bid']['size']})")
        print(f"💸 Best Ask: ${data['best_ask']['price']} (Size: {data['best_ask']['size']})")
        
        if data['spread']:
            print(f"📏 Spread: ${data['spread']['absolute']} ({data['spread']['percentage']}%)")
            print(f"🎯 Mid Price: ${data['spread']['mid_price']}")
    
    # Show changes (delta updates)
    if data['changes']:
        print(f"🔄 Changes ({len(data['changes'])} updates):")
        for change in data['changes'][:5]:  # Show first 5 changes
            side, price, size = change
            action = "REMOVE" if float(size) == 0 else "UPDATE"
            print(f"   {side.upper()} {action}: ${price} -> {size}")
        
        if len(data['changes']) > 5:
            print(f"   ... and {len(data['changes']) - 5} more changes")
    
    # Show top levels summary
    if data['top_bids'] and data['top_asks']:
        print(f"📋 Order Book Depth: {len(data['top_bids'])} bids, {len(data['top_asks'])} asks")
        
        # Show top 3 levels
        print("🔝 Top 3 Bids:")
        for i, (price, size) in enumerate(data['top_bids'][:3]):
            print(f"   {i+1}. ${price} -> {size}")
            
        print("🔝 Top 3 Asks:")
        for i, (price, size) in enumerate(data['top_asks'][:3]):
            print(f"   {i+1}. ${price} -> {size}")
    
    print("="*60)

def test_market_feed():
    """Test the MarketDataFeed with ETH-USD"""
    print("🚀 Testing Coinbase WebSocket MarketDataFeed")
    print("=" * 50)
    
    try:
        # Initialize MarketDataFeed
        print("1. Initializing MarketDataFeed...")
        feed = MarketDataFeed(environment='production')
        print("✅ MarketDataFeed initialized")
        
        # Start listening to ETH-USD
        print("\n2. Starting ETH-USD Level 2 order book stream...")
        feed.listen_to_data('DOGE-USD', market_data_callback)
        print("✅ Started listening to DOGE-USD")
        
        print("\n3. Streaming market data (press Ctrl+C to stop)...")
        print("📡 Waiting for Coinbase WebSocket connection...")
        
        # Let it run for a while to collect data
        try:
            while True:
                time.sleep(1)
                
                # Optionally show current state every 30 seconds
                if int(time.time()) % 30 == 0:
                    current_state = feed.get_current_order_book('DOGE-USD')
                    if current_state:
                        print(f"\n📊 Current Order Book State:")
                        print(f"   Best Bid: ${current_state['best_bid']['price']}" if current_state['best_bid'] else "   No bids")
                        print(f"   Best Ask: ${current_state['best_ask']['price']}" if current_state['best_ask'] else "   No asks")
                        print(f"   Book Depth: {len(current_state['top_bids'])} bids, {len(current_state['top_asks'])} asks")
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping market data feed...")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    finally:
        # Clean shutdown
        if 'feed' in locals():
            feed.stop_feed()
            print("✅ Market data feed stopped cleanly")
    
    return True

def test_signal_generator_integration():
    """
    Demonstrate how this integrates with SignalGenerator
    """
    print("\n" + "="*60)
    print("🔗 SignalGenerator Integration Example")
    print("="*60)
    
    print("""
    Integration with your existing SignalGenerator:
    
    1. 📊 MarketDataFeed streams Level 2 order book data
    2. 🧠 SignalGenerator.refresh() receives formatted data:
       - Delta changes for momentum analysis
       - Best bid/ask for pricing decisions  
       - Order book depth for liquidity assessment
       - Spread information for market conditions
    
    3. 🎯 SignalGenerator can now use:
       - data['best_bid']['price'] for current market price
       - data['changes'] for order flow analysis
       - data['spread']['percentage'] for volatility assessment
       - data['top_bids']/data['top_asks'] for depth analysis
    
    Example SignalGenerator.refresh() usage:
    
    def refresh(self, market_data):
        # Get current price from best bid/ask
        if market_data['best_bid'] and market_data['best_ask']:
            current_price = (
                float(market_data['best_bid']['price']) + 
                float(market_data['best_ask']['price'])
            ) / 2
            
        # Analyze order flow from changes
        buy_pressure = sum(float(size) for side, price, size in market_data['changes'] 
                          if side == 'buy' and float(size) > 0)
        
        # Check spread for market conditions
        spread_pct = market_data['spread']['percentage']
        tight_spread = spread_pct < 0.05  # Less than 0.05%
        
        # Generate signals based on this data
        signals = self._target_model.predict({
            'price': current_price,
            'buy_pressure': buy_pressure,
            'spread': spread_pct,
            'tight_market': tight_spread
        })
        
        return signals
    """)

if __name__ == "__main__":
    load_dotenv()
    
    print("Coinbase WebSocket MarketDataFeed Test")
    print("=" * 50)
    print("⚠️  Make sure you have COINBASE_API_KEY and COINBASE_API_SECRET in your .env file")
    print("📝 Note: This uses Coinbase's public WebSocket feed (no auth required for market data)")
    print()
    
    # Run the test
    success = test_market_feed()
    
    # Show integration example
    test_signal_generator_integration()
    
    print("\n" + "="*50)
    print("📝 Next Steps:")
    print("1. Update your SignalGenerator to use the new data format")
    print("2. Test with your ML models using the rich order book data")
    print("3. Implement trading strategies based on Level 2 data")
    print("4. Monitor performance and adjust order book depth as needed")
    print("="*50)
    
    sys.exit(0 if success else 1)
