#!/usr/bin/env python3
"""
Simple test to verify the TradingClient import works correctly
"""

import sys
sys.path.append('src')

try:
    from trading_client.trading_client import TradingClient
    print("✅ SUCCESS: TradingClient import works correctly!")
    print("✅ SUCCESS: Coinbase Advanced Trade API library is properly installed")
    print("\nThe import error has been fixed. You can now:")
    print("1. Create a .env file with your Coinbase sandbox credentials")
    print("2. Run: python test_trading_client.py")
    print("3. Start using the TradingClient in your trading bot")
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("Please check that coinbase-advanced-py is installed:")
    print("pip install coinbase-advanced-py")
except Exception as e:
    print(f"❌ OTHER ERROR: {e}")
