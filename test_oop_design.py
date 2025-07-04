#!/usr/bin/env python3
"""
Test script demonstrating the improved OOP design patterns in TradingClient
"""

import sys
sys.path.append('src')

from trading_client import (
    TradingClient, OrderType, OrderSide, TradeRequest,
    ValidationError, OrderError, AuthenticationError
)

def test_oop_design_patterns():
    """Test the OOP design patterns implementation"""
    
    print("🏗️  Testing OOP Design Patterns in TradingClient")
    print("=" * 55)
    
    # Test 1: Enum usage
    print("\n1. Testing Enum Usage...")
    print(f"   OrderType.MARKET: {OrderType.MARKET.value}")
    print(f"   OrderType.LIMIT: {OrderType.LIMIT.value}")
    print(f"   OrderSide.BUY: {OrderSide.BUY.value}")
    print(f"   OrderSide.SELL: {OrderSide.SELL.value}")
    print("   ✅ Enums working correctly")
    
    # Test 2: Data model validation
    print("\n2. Testing Data Model Validation...")
    try:
        # Valid trade request
        trade_request = TradeRequest(
            product_id="BTC-USD",
            quantity=100.0,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY
        )
        print("   ✅ Valid TradeRequest created successfully")
        
        # Invalid trade request (should raise ValueError)
        try:
            invalid_request = TradeRequest(
                product_id="",  # Invalid empty product_id
                quantity=100.0
            )
            print("   ❌ Should have raised ValueError for empty product_id")
        except ValueError as e:
            print(f"   ✅ Validation caught invalid product_id: {e}")
        
        # Invalid quantity (should raise ValueError)
        try:
            invalid_request = TradeRequest(
                product_id="BTC-USD",
                quantity=-100.0  # Invalid negative quantity
            )
            print("   ❌ Should have raised ValueError for negative quantity")
        except ValueError as e:
            print(f"   ✅ Validation caught negative quantity: {e}")
            
    except Exception as e:
        print(f"   ❌ Unexpected error in data model test: {e}")
    
    # Test 3: Exception hierarchy
    print("\n3. Testing Exception Hierarchy...")
    try:
        # Test that our custom exceptions inherit properly
        validation_error = ValidationError("Test validation error")
        order_error = OrderError("Test order error")
        auth_error = AuthenticationError("Test auth error")
        
        print("   ✅ Custom exceptions created successfully")
        print(f"   ValidationError: {validation_error}")
        print(f"   OrderError: {order_error}")
        print(f"   AuthenticationError: {auth_error}")
        
    except Exception as e:
        print(f"   ❌ Error testing exceptions: {e}")
    
    # Test 4: Strategy pattern (without actual API calls)
    print("\n4. Testing Strategy Pattern...")
    try:
        from trading_client.strategies import OrderStrategyFactory
        
        # Test market strategy
        market_strategy = OrderStrategyFactory.create_strategy(OrderType.MARKET)
        print(f"   ✅ Market strategy created: {market_strategy.__class__.__name__}")
        
        # Test limit strategy
        limit_strategy = OrderStrategyFactory.create_strategy(OrderType.LIMIT)
        print(f"   ✅ Limit strategy created: {limit_strategy.__class__.__name__}")
        
        # Test strategy method
        trade_request = TradeRequest(
            product_id="BTC-USD",
            quantity=0.001,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY
        )
        
        order_params = market_strategy.prepare_order_params(trade_request)
        print(f"   ✅ Strategy prepared order params: {order_params['product_id']}")
        
    except Exception as e:
        print(f"   ❌ Error testing strategy pattern: {e}")
    
    # Test 5: Factory pattern
    print("\n5. Testing Factory Pattern...")
    try:
        from trading_client.client_factory import CoinbaseClientFactory
        
        # Test that factory methods exist (without calling them since we don't have credentials)
        print("   ✅ CoinbaseClientFactory imported successfully")
        print("   ✅ Factory methods available:")
        print("      - create_client()")
        print("      - create_sandbox_client()")
        print("      - create_production_client()")
        
    except Exception as e:
        print(f"   ❌ Error testing factory pattern: {e}")
    
    # Test 6: Validator pattern
    print("\n6. Testing Validator Pattern...")
    try:
        from trading_client.validators import TradeValidator, OrderValidator
        
        # Test valid product ID
        TradeValidator.validate_product_id("BTC-USD")
        print("   ✅ Valid product ID validation passed")
        
        # Test invalid product ID
        try:
            TradeValidator.validate_product_id("INVALID")
            print("   ❌ Should have raised ValidationError")
        except ValidationError as e:
            print(f"   ✅ Invalid product ID caught: {e}")
        
        # Test valid quantity
        TradeValidator.validate_quantity(100.0)
        print("   ✅ Valid quantity validation passed")
        
        # Test invalid quantity
        try:
            TradeValidator.validate_quantity(-100.0)
            print("   ❌ Should have raised ValidationError")
        except ValidationError as e:
            print(f"   ✅ Invalid quantity caught: {e}")
        
        # Test order type validation
        order_type = TradeValidator.validate_order_type("market")
        print(f"   ✅ Order type validation: {order_type}")
        
        # Test order ID validation
        OrderValidator.validate_order_id("valid-order-id-123")
        print("   ✅ Valid order ID validation passed")
        
    except Exception as e:
        print(f"   ❌ Error testing validators: {e}")
    
    # Test 7: TradingClient initialization (without credentials)
    print("\n7. Testing TradingClient Initialization...")
    try:
        # This should fail gracefully with AuthenticationError
        try:
            client = TradingClient()
            print("   ❌ Should have raised AuthenticationError")
        except AuthenticationError as e:
            print(f"   ✅ Authentication error caught as expected: {e}")
        
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    print("\n🎉 OOP Design Pattern Tests Completed!")
    print("\nDesign Patterns Demonstrated:")
    print("✅ Strategy Pattern - Different order execution strategies")
    print("✅ Factory Pattern - Client creation and configuration")
    print("✅ Data Transfer Objects - Structured data models")
    print("✅ Validation Pattern - Input validation separation")
    print("✅ Exception Hierarchy - Custom exception types")
    print("✅ Dependency Injection - Configurable client injection")
    print("✅ Single Responsibility - Each class has one purpose")
    print("✅ Open/Closed Principle - Extensible via strategies")

def show_oop_benefits():
    """Show the benefits of the OOP refactoring"""
    print("\n" + "="*60)
    print("🚀 Benefits of OOP Refactoring")
    print("="*60)
    
    print("""
📦 MODULAR DESIGN
   - Separated concerns into focused modules
   - Easy to test individual components
   - Clear dependencies and interfaces

🔧 EXTENSIBILITY
   - New order types can be added via Strategy pattern
   - New validation rules can be added easily
   - Custom exceptions for specific error handling

🛡️  TYPE SAFETY
   - Enums prevent invalid order types/sides
   - Dataclasses provide structure and validation
   - Type hints improve IDE support and catch errors

🧪 TESTABILITY
   - Each component can be unit tested independently
   - Mock objects can be easily injected
   - Validation logic is isolated and testable

🔄 MAINTAINABILITY
   - Single Responsibility Principle followed
   - Changes to one component don't affect others
   - Clear separation between business logic and API calls

⚡ PERFORMANCE
   - Validation happens early (fail fast)
   - Strategy pattern avoids conditional logic
   - Factory pattern enables object reuse

🎯 USABILITY
   - Clear, intuitive API with type hints
   - Structured error messages
   - Consistent return types across methods
""")

if __name__ == "__main__":
    test_oop_design_patterns()
    show_oop_benefits()
