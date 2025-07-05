# Trading Bot Project

A comprehensive cryptocurrency trading bot with machine learning capabilities, real-time market data processing, and automated trading strategies.

## Project Structure

```
trading/
├── src/                    # Core source code
│   ├── data/              # Data processing and generation
│   ├── market_feed/       # Real-time market data feeds
│   ├── models/            # Machine learning models
│   ├── model_trainer/     # Model training utilities
│   ├── model_registry/    # Model management and versioning
│   ├── orchestrator/      # Workflow orchestration
│   ├── order_manager/     # Trade execution management
│   ├── signal_generator/  # Trading signal generation
│   └── trading_client/    # Trading client implementations
├── tests/                 # Test files
├── demos/                 # Demo scripts and examples
├── docs/                  # Documentation files
├── data/                  # Historical data storage
├── models/                # Trained model storage
├── demo_trading_data/     # Demo data files
├── demo_trading_models/   # Demo model files
├── catboost_info/         # CatBoost training logs
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── .gitignore            # Git ignore rules
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```

3. **Run Tests**
   ```bash
   python tests/test_workflow_orchestrator.py
   ```

4. **Run Demos**
   ```bash
   python demos/demo_trading_flow.py
   python demos/demo_workflow_orchestrator.py
   ```

## Key Features

- **Real-time Market Data**: Live cryptocurrency price feeds from Coinbase
- **Machine Learning Models**: CatBoost and heuristic trading models
- **Automated Trading**: Signal generation and order execution
- **Risk Management**: Position sizing and trade limits
- **Workflow Orchestration**: Scheduled data collection and model training
- **Comprehensive Testing**: Full test suite for all components

## Architecture

The trading bot follows a modular architecture with clear separation of concerns:

- **Data Layer**: Market data collection and historical data management
- **Model Layer**: Machine learning models for price prediction
- **Signal Layer**: Trading signal generation based on model predictions
- **Execution Layer**: Order management and trade execution
- **Orchestration Layer**: Workflow coordination and scheduling

## Documentation

Detailed documentation can be found in the `docs/` directory:

- [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)
- [Workflow Orchestrator](docs/WORKFLOW_ORCHESTRATOR_IMPLEMENTATION_SUMMARY.md)
- [Order Manager](docs/ORDER_MANAGER_IMPLEMENTATION_SUMMARY.md)
- [Signal Generator](docs/SIGNAL_GENERATOR_FIX_SUMMARY.md)
- [Coinbase WebSocket](docs/COINBASE_WEBSOCKET_IMPLEMENTATION.md)

## Safety Features

- **Sandbox Mode**: Safe testing environment using Coinbase sandbox
- **Environment Isolation**: Separate configurations for development and production
- **Comprehensive Logging**: Detailed logging for monitoring and debugging
- **Error Handling**: Robust error handling and recovery mechanisms

## Contributing

1. Run tests before submitting changes
2. Follow the existing code structure and patterns
3. Update documentation for new features
4. Use the demo scripts to validate functionality

## License

This project is for educational and research purposes.
