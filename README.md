# Trading Bot with Coinbase Advanced Trade API

A comprehensive trading bot implementation with Coinbase Advanced Trade API integration for cryptocurrency trading.

## 🚀 Features

- **TradingClient**: Full Coinbase Advanced Trade API integration
- **Market Orders**: Execute buy/sell orders at market price
- **Limit Orders**: Place orders at specific price levels
- **Portfolio Management**: View balances, positions, and holdings
- **Order Management**: Track, cancel, and monitor order status
- **Sandbox Support**: Safe testing environment
- **Error Handling**: Comprehensive error handling and logging

## 📋 Prerequisites

- Python 3.7+
- Coinbase Advanced Trade API credentials
- Access to Coinbase Pro/Advanced Trade (sandbox for testing)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd trading_bot2
   ```

2. **Install dependencies**
   ```bash
   pip install coinbase-advanced-py python-dotenv
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` file with your Coinbase credentials:
   ```env
   COINBASE_API_KEY=your_api_key_here
   COINBASE_API_SECRET=your_api_secret_here
   COINBASE_ENVIRONMENT=sandbox
   ```

## 🔑 Getting Coinbase API Credentials

1. Go to [Coinbase Advanced Trade](https://pro.coinbase.com/) (or sandbox)
2. Navigate to API settings
3. Create a new API key with trading permissions
4. Save the API Key, Secret, and Passphrase securely

## 🧪 Testing

Run the test script to verify your setup:

```bash
python test_trading_client.py
```

This will test all TradingClient methods without placing actual trades (trading tests are commented out for safety).

## 📖 Usage Examples

### Basic Usage

```python
from src.trading_client.trading_client import TradingClient

# Initialize client (uses .env credentials)
client = TradingClient()

# Get account balances
balances = client.get_account_balance()
print(f"USD Balance: ${balances['balances']['USD']['available']}")

# View ETH position
position = client.view_position('ETH-USD')
print(f"ETH Holdings: {position['quantity_held']} ETH")
print(f"Current Price: ${position['current_price']}")
```

### Trading Operations

```python
# Execute a market buy order (buy $50 worth of BTC)
trade_result = client.execute_trade(
    product_id='BTC-USD',
    quantity=50,  # $50 USD
    order_type='market',
    side='buy'
)

if trade_result['success']:
    print(f"Order ID: {trade_result['order_id']}")
    
    # Check order status
    status = client.get_trade_status(trade_result['order_id'])
    print(f"Order Status: {status['status']}")
    print(f"Filled: {status['filled_size']}")
    
    # Cancel order if still pending
    if status['status'] in ['pending', 'open']:
        cancel_result = client.cancel_trade(trade_result['order_id'])
        print(f"Cancel Status: {cancel_result['message']}")
else:
    print(f"Trade failed: {trade_result['error']}")

# Execute a limit sell order (sell 0.01 ETH at $3000)
trade_result = client.execute_trade(
    product_id='ETH-USD',
    quantity=0.01,  # 0.01 ETH
    price=3000.00,  # At $3000 per ETH
    order_type='limit',
    side='sell'
)
```

### Portfolio Management

```python
# Get complete portfolio
portfolio = client.get_portfolio()
for asset, details in portfolio['holdings'].items():
    print(f"{asset}: {details['quantity']} (Available: {details['available']})")

# Get detailed account balances
balances = client.get_account_balance()
for currency, details in balances['balances'].items():
    if details['balance'] > 0:
        print(f"{currency}: {details['balance']} (Hold: {details['hold']})")
```

## 🏗️ Architecture

The trading bot consists of several components:

- **TradingClient** (`src/trading_client/`): Coinbase API integration
- **OrderManager** (`src/order_manager/`): Order execution and management
- **SignalGenerator** (`src/signal_generator/`): Trading signal generation
- **MarketDataFeed** (`src/market_feed/`): Market data streaming
- **ModelRegistry** (`src/model_registry/`): ML model management
- **WorkflowOrchestrator** (`src/orchestrator/`): Workflow coordination

## 🔧 TradingClient API Reference

### Implemented Methods

#### `execute_trade(product_id, quantity, price=None, order_type='market', side=None)`
Execute a trade order using Coinbase Advanced Trade API.

**Parameters:**
- `product_id` (str, required): Trading pair (e.g., 'BTC-USD', 'ETH-USD')
- `quantity` (float, required): Amount to trade (positive number)
- `price` (float, optional): Price for limit orders (required for limit orders)
- `order_type` (str, optional): 'market' or 'limit'. Defaults to 'market'
- `side` (str, optional): 'buy' or 'sell' (auto-determined if not provided)

**Returns:** Dict with success status, order ID, and execution details
```python
{
    'success': True,
    'order_id': 'abc123...',
    'product_id': 'BTC-USD',
    'side': 'buy',
    'order_type': 'market',
    'status': 'pending',
    'created_time': '2024-01-01T12:00:00Z'
}
```

#### `cancel_trade(order_id)`
Cancel a pending order.

**Parameters:**
- `order_id` (str): Order ID to cancel

**Returns:** Dict with cancellation confirmation
```python
{
    'success': True,
    'order_id': 'abc123...',
    'status': 'cancelled',
    'message': 'Order cancelled successfully'
}
```

#### `get_trade_status(order_id)`
Get detailed order status and fill information.

**Parameters:**
- `order_id` (str): Order ID to check

**Returns:** Dict with detailed order status
```python
{
    'order_id': 'abc123...',
    'product_id': 'BTC-USD',
    'side': 'buy',
    'status': 'filled',
    'size': '0.001',
    'filled_size': '0.001',
    'average_filled_price': '45000.00',
    'created_time': '2024-01-01T12:00:00Z'
}
```

#### `get_account_balance()`
Get account balances for all currencies.

**Returns:** Dict with balance information for each currency
```python
{
    'balances': {
        'USD': {
            'balance': 1000.00,
            'available': 950.00,
            'hold': 50.00,
            'currency': 'USD'
        },
        'BTC': {
            'balance': 0.1,
            'available': 0.1,
            'hold': 0.0,
            'currency': 'BTC'
        }
    },
    'total_value_usd': 1000.00,
    'timestamp': '2024-01-01T12:00:00Z'
}
```

### Legacy Methods (Not Yet Implemented)

#### `get_portfolio()`
Get portfolio holdings (assets with non-zero balance).

**Status:** Placeholder - not yet implemented

#### `view_position(product)`
Get detailed position information for a specific product.

**Status:** Placeholder - not yet implemented

## ⚠️ Safety Notes

1. **Always test in sandbox first** before using production credentials
2. **Start with small amounts** when testing real trades
3. **Monitor your trades** and set appropriate stop-losses
4. **Keep API credentials secure** and never commit them to version control
5. **Respect rate limits** (10 requests/second for sandbox, 10 requests/second for production)

## 🐛 Troubleshooting

### Common Issues

1. **Authentication Error**
   - Verify API credentials in `.env` file
   - Ensure API key has trading permissions
   - Check if using correct environment (sandbox vs production)

2. **Import Errors**
   - Make sure all dependencies are installed: `pip install coinbase-advanced-py python-dotenv`
   - Verify Python path includes the `src` directory

3. **API Rate Limits**
   - Implement delays between requests
   - Use the built-in rate limiting in the client

4. **Order Errors**
   - Check account balance before placing orders
   - Verify product ID format (e.g., 'ETH-USD' not 'ETHUSD')
   - Ensure minimum order sizes are met

## 📝 Logging

The TradingClient includes comprehensive logging. Logs include:
- Order executions and confirmations
- API errors and responses
- Balance and portfolio updates
- Authentication status

## 🔮 Future Enhancements

- WebSocket integration for real-time market data
- Advanced order types (stop-loss, take-profit)
- Portfolio rebalancing algorithms
- Risk management features
- Backtesting capabilities
- Performance analytics

## 📄 License

This project is for educational and personal use. Please ensure compliance with Coinbase's API terms of service.
