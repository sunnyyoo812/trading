"""
Workflow Orchestrator for managing automated training pipeline with scheduling
"""

import os
import logging
from datetime import datetime, time
from typing import Dict, Optional, List
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.data.historical_market_data_generator import HistoricalMarketDataGenerator
from src.model_trainer.model_trainer import ModelTrainer
from src.model_registry.model_registry import ModelRegistry
from src.models.target_model import TargetModel
from src.market_feed.market_feed import MarketDataFeed
from src.signal_generator.signal_generator import SignalGenerator
from src.order_manager.order_manager import OrderManager
from src.trading_client.trading_client import TradingClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """
    Orchestrates the complete machine learning workflow including:
    - Daily data collection and storage (4 PM EST)
    - Daily model training and deployment (5 PM EST)
    - Model registry management
    - Error handling and recovery
    """
    
    def __init__(self, 
                 data_dir: str = "data/historical",
                 model_dir: str = "models",
                 symbols: List[str] = None,
                 model_type: str = 'catboost'):
        """
        Initialize the workflow orchestrator
        
        Parameters:
        - data_dir: Directory for storing historical data
        - model_dir: Directory for storing trained models
        - symbols: List of trading symbols to process
        - model_type: Type of model to train ('catboost', 'heuristic')
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.symbols = symbols or ['DOGE-USD', 'BTC-USD']
        self.model_type = model_type
        
        # Initialize components
        self.data_generator = HistoricalMarketDataGenerator(data_dir=data_dir)
        self.model_trainer = ModelTrainer(model_type=model_type)
        self.model_registry = ModelRegistry()
        
        # Scheduler for automated tasks
        self.scheduler = BackgroundScheduler()
        self.est_tz = pytz.timezone('US/Eastern')
        
        # State tracking
        self.is_running = False
        self.last_data_save = None
        self.last_training = None
        self.current_model = None
        
        # Trading components
        self.trading_market_feed = None
        self.trading_signal_generator = None
        self.trading_client = None
        self.trading_order_manager = None
        self.is_trading = False
        self.trading_environment = 'sandbox'
        self.trade_amount_usd = 1.0
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info(f"WorkflowOrchestrator initialized for symbols: {self.symbols}")
    
    def start_scheduler(self):
        """
        Start the automated workflow scheduler
        
        Schedules:
        - 4:00 PM EST: Save historical data
        - 5:00 PM EST: Train and deploy model
        """
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        
        # Schedule data saving at 4:00 PM EST daily
        self.scheduler.add_job(
            func=self.collect_and_save_historical_data,
            trigger=CronTrigger(hour=16, minute=0, timezone=self.est_tz),
            id='daily_data_save',
            name='Daily Historical Data Save',
            replace_existing=True
        )
        
        # Schedule model training at 5:00 PM EST daily
        self.scheduler.add_job(
            func=self.run_training_pipeline,
            trigger=CronTrigger(hour=17, minute=0, timezone=self.est_tz),
            id='daily_training',
            name='Daily Model Training',
            replace_existing=True
        )
        
        # Start data collection immediately
        self.data_generator.start_data_collection(self.symbols)
        
        # Start scheduler
        self.scheduler.start()
        self.is_running = True
        
        logger.info("Workflow scheduler started:")
        logger.info("  - Data save: Daily at 4:00 PM EST")
        logger.info("  - Model training: Daily at 5:00 PM EST")
        logger.info(f"  - Data collection: Started for {self.symbols}")
    
    def collect_and_save_historical_data(self) -> Dict[str, str]:
        """
        Collect and save historical data (scheduled for 4:00 PM EST)
        
        Returns:
        - Dictionary mapping symbols to saved file paths
        """
        logger.info("🕐 Starting scheduled data collection and save...")
        
        try:
            # Save collected data
            saved_files = self.data_generator.save_daily_historical_data()
            
            if saved_files:
                self.last_data_save = datetime.now(self.est_tz)
                logger.info(f"✅ Data saved successfully for {len(saved_files)} symbols")
                for symbol, filepath in saved_files.items():
                    logger.info(f"   {symbol}: {filepath}")
            else:
                logger.warning("⚠️ No data was saved - check data collection")
            
            return saved_files
            
        except Exception as e:
            logger.error(f"❌ Error during data save: {e}")
            return {}
    
    def run_training_pipeline(self) -> Optional[TargetModel]:
        """
        Run the complete training pipeline (scheduled for 5:00 PM EST)
        
        Returns:
        - Trained model if successful, None otherwise
        """
        logger.info("🚀 Starting scheduled training pipeline...")
        
        try:
            # Load historical data
            df = self.load_historical_data()
            
            if df.empty:
                logger.error("❌ No historical data available for training")
                return None
            
            # Train model
            model = self.train_model(df)
            
            if model is None:
                logger.error("❌ Model training failed")
                return None
            
            # Evaluate model
            metrics = self.evaluate_model(model, df)
            
            # Save and deploy model
            model_path = self.deploy_model(model, metrics)
            
            self.last_training = datetime.now(self.est_tz)
            self.current_model = model
            
            logger.info("✅ Training pipeline completed successfully")
            logger.info(f"   Model saved to: {model_path}")
            logger.info(f"   RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Error during training pipeline: {e}")
            return None
    
    def load_historical_data(self) -> 'pd.DataFrame':
        """
        Load historical data for training
        
        Returns:
        - DataFrame with historical data from last 7 days
        """
        logger.info("Loading historical data for training...")
        
        df = self.data_generator.load_historical_data(
            symbols=self.symbols,
            days_back=7
        )
        
        if not df.empty:
            logger.info(f"Loaded {len(df)} records for training")
        else:
            logger.warning("No historical data found")
        
        return df
    
    def train_model(self, df: 'pd.DataFrame') -> Optional[TargetModel]:
        """
        Train a model using historical data
        
        Parameters:
        - df: Historical data for training
        
        Returns:
        - Trained model or None if training fails
        """
        logger.info(f"Training {self.model_type} model...")
        
        try:
            model = self.model_trainer.train_model(df)
            logger.info("Model training completed successfully")
            return model
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return None
    
    def evaluate_model(self, model: TargetModel, df: 'pd.DataFrame') -> Dict:
        """
        Evaluate trained model
        
        Parameters:
        - model: Trained model to evaluate
        - df: Data for evaluation
        
        Returns:
        - Dictionary of evaluation metrics
        """
        logger.info("Evaluating trained model...")
        
        try:
            metrics = self.model_trainer.evaluate_model(df)
            
            # Log metrics
            self.model_trainer.log_model(model, metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {}
    
    def deploy_model(self, model: TargetModel, metrics: Dict) -> str:
        """
        Deploy trained model to registry and save to file
        
        Parameters:
        - model: Trained model to deploy
        - metrics: Model evaluation metrics
        
        Returns:
        - Path to saved model file
        """
        logger.info("Deploying trained model...")
        
        # Generate model filename with timestamp
        timestamp = datetime.now(self.est_tz).strftime('%Y%m%d_%H%M%S')
        model_filename = f"{self.model_type}_model_{timestamp}.pkl"
        model_path = os.path.join(self.model_dir, model_filename)
        
        try:
            # Save model to file
            self.model_trainer.save_model(model, model_path)
            
            # Register model in registry
            model_name = f"{self.model_type}_latest"
            
            # Unregister previous model if exists
            try:
                self.model_registry.unregister_model(model_name)
            except ValueError:
                pass  # Model doesn't exist yet
            
            # Register new model
            self.model_registry.register_model(model_name, model)
            
            logger.info(f"Model deployed successfully as '{model_name}'")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            raise
    
    def get_current_model(self) -> Optional[TargetModel]:
        """
        Get the currently deployed model
        
        Returns:
        - Current model or None if no model is deployed
        """
        try:
            return self.model_registry.get_model(f"{self.model_type}_latest")
        except ValueError:
            return None
    
    def manual_training(self) -> Optional[TargetModel]:
        """
        Manually trigger training pipeline (for testing/debugging)
        
        Returns:
        - Trained model if successful
        """
        logger.info("Manual training triggered...")
        return self.run_training_pipeline()
    
    def manual_data_save(self) -> Dict[str, str]:
        """
        Manually trigger data save (for testing/debugging)
        
        Returns:
        - Dictionary of saved files
        """
        logger.info("Manual data save triggered...")
        return self.collect_and_save_historical_data()
    
    def get_status(self) -> Dict:
        """
        Get current workflow status
        
        Returns:
        - Dictionary with workflow status information
        """
        # Get data collection status
        data_status = self.data_generator.get_collection_status()
        
        # Get scheduler status
        scheduler_jobs = []
        if self.scheduler.running:
            for job in self.scheduler.get_jobs():
                scheduler_jobs.append({
                    'id': job.id,
                    'name': job.name,
                    'next_run': job.next_run_time.isoformat() if job.next_run_time else None
                })
        
        # Get current model info
        current_model_info = None
        if self.current_model and hasattr(self.current_model, 'get_model_info'):
            current_model_info = self.current_model.get_model_info()
        
        return {
            'is_running': self.is_running,
            'scheduler_running': self.scheduler.running if hasattr(self.scheduler, 'running') else False,
            'symbols': self.symbols,
            'model_type': self.model_type,
            'last_data_save': self.last_data_save.isoformat() if self.last_data_save else None,
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'data_collection': data_status,
            'scheduled_jobs': scheduler_jobs,
            'current_model': current_model_info,
            'registered_models': self.model_registry.list_models()
        }
    
    def start_trading_flow(self, environment: str = 'sandbox', trade_amount: float = 100.0, 
                          buy_threshold: float = 0.5, sell_threshold: float = 0.5) -> bool:
        """
        Start the complete trading flow
        
        Parameters:
        - environment: Trading environment ('sandbox' or 'production')
        - trade_amount: USD amount per trade
        - buy_threshold: Signal threshold for buy orders (%)
        - sell_threshold: Signal threshold for sell orders (%)
        
        Returns:
        - True if trading flow started successfully, False otherwise
        """
        if self.is_trading:
            logger.warning("Trading flow already running")
            return True
        
        logger.info("🚀 Starting trading flow...")
        logger.info(f"   Environment: {environment}")
        logger.info(f"   Trade Amount: ${trade_amount}")
        logger.info(f"   Symbols: {self.symbols}")
        
        try:
            # 1. Load latest model from registry
            current_model = self.get_current_model()
            if current_model is None:
                logger.error("❌ No trained model available - run training first")
                return False
            
            logger.info(f"✅ Loaded model: {current_model.get_model_info()['model_type']}")
            
            # 2. Initialize trading components
            self.trading_environment = environment
            self.trade_amount_usd = trade_amount
            
            # Create market data feed for trading (separate from training data collection)
            self.trading_market_feed = MarketDataFeed(environment=environment)
            logger.info("✅ Trading market feed initialized")
            
            # Create signal generator with the trained model
            self.trading_signal_generator = SignalGenerator(
                target_model=current_model,
                market_data_feed=self.trading_market_feed,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold
            )
            logger.info("✅ Signal generator initialized")
            
            # Create trading client
            self.trading_client = TradingClient(environment=environment)
            logger.info(f"✅ Trading client initialized ({environment} mode)")
            
            # Create order manager and subscribe to signals
            self.trading_order_manager = OrderManager(
                trading_client=self.trading_client,
                trade_amount_usd=trade_amount
            )
            logger.info("✅ Order manager initialized")
            
            # 3. Connect the pipeline: SignalGenerator -> OrderManager
            self.trading_signal_generator.subscribers = [self.trading_order_manager]
            logger.info("✅ Trading pipeline connected")
            
            # 4. Start market data feeds for all symbols
            for symbol in self.symbols:
                self.trading_market_feed.listen_to_data(symbol, self.trading_signal_generator.refresh)
                logger.info(f"✅ Started market data for {symbol}")
            
            self.is_trading = True
            logger.info("🎯 Trading flow started successfully!")
            logger.info("   Market Data → Signal Generator → Order Manager → Trading Client")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start trading flow: {e}")
            self.stop_trading_flow()
            return False
    
    def stop_trading_flow(self):
        """
        Stop the trading flow and cleanup all trading components
        """
        if not self.is_trading:
            logger.info("Trading flow not running")
            return
        
        logger.info("🛑 Stopping trading flow...")
        
        try:
            # Stop market data feed
            if self.trading_market_feed:
                self.trading_market_feed.stop_feed()
                self.trading_market_feed = None
                logger.info("✅ Trading market feed stopped")
            
            # Clear signal generator
            if self.trading_signal_generator:
                self.trading_signal_generator = None
                logger.info("✅ Signal generator cleared")
            
            # Clear order manager
            if self.trading_order_manager:
                self.trading_order_manager = None
                logger.info("✅ Order manager cleared")
            
            # Clear trading client
            if self.trading_client:
                self.trading_client = None
                logger.info("✅ Trading client cleared")
            
            self.is_trading = False
            logger.info("✅ Trading flow stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping trading flow: {e}")
            self.is_trading = False
    
    def restart_trading_flow(self, **kwargs):
        """
        Restart the trading flow with optional new parameters
        
        Parameters:
        - **kwargs: Parameters to pass to start_trading_flow
        """
        logger.info("🔄 Restarting trading flow...")
        self.stop_trading_flow()
        
        # Use current settings as defaults
        params = {
            'environment': self.trading_environment,
            'trade_amount': self.trade_amount_usd
        }
        params.update(kwargs)
        
        return self.start_trading_flow(**params)
    
    def get_trading_status(self) -> Dict:
        """
        Get current trading flow status
        
        Returns:
        - Dictionary with trading status information
        """
        status = {
            'is_trading': self.is_trading,
            'environment': self.trading_environment,
            'trade_amount_usd': self.trade_amount_usd,
            'symbols': self.symbols,
            'components': {
                'market_feed': self.trading_market_feed is not None,
                'signal_generator': self.trading_signal_generator is not None,
                'trading_client': self.trading_client is not None,
                'order_manager': self.trading_order_manager is not None
            }
        }
        
        # Add model information
        current_model = self.get_current_model()
        if current_model:
            status['current_model'] = current_model.get_model_info()
        else:
            status['current_model'] = None
        
        # Add order manager status if available
        if self.trading_order_manager:
            try:
                status['order_manager_status'] = {
                    'total_trades': len(getattr(self.trading_order_manager.trading_client, 'executed_trades', [])),
                    'trade_amount': self.trading_order_manager.trade_amount_usd
                }
            except:
                status['order_manager_status'] = 'unavailable'
        
        return status
    
    def get_trading_performance(self) -> Dict:
        """
        Get trading performance metrics
        
        Returns:
        - Dictionary with performance information
        """
        if not self.is_trading or not self.trading_order_manager:
            return {'error': 'Trading not active'}
        
        try:
            # Get executed trades from trading client
            executed_trades = getattr(self.trading_order_manager.trading_client, 'executed_trades', [])
            
            performance = {
                'total_trades': len(executed_trades),
                'trades_by_symbol': {},
                'trades_by_side': {'buy': 0, 'sell': 0},
                'total_volume': 0.0
            }
            
            for trade in executed_trades:
                symbol = trade.get('symbol', 'unknown')
                side = trade.get('side', 'unknown')
                size = float(trade.get('size', 0))
                
                # Count by symbol
                if symbol not in performance['trades_by_symbol']:
                    performance['trades_by_symbol'][symbol] = 0
                performance['trades_by_symbol'][symbol] += 1
                
                # Count by side
                if side in performance['trades_by_side']:
                    performance['trades_by_side'][side] += 1
                
                # Add to volume
                performance['total_volume'] += size
            
            return performance
            
        except Exception as e:
            return {'error': f'Failed to get performance: {e}'}
    
    def update_trading_config(self, **kwargs):
        """
        Update trading configuration
        
        Parameters:
        - **kwargs: Configuration parameters to update
        """
        if 'trade_amount' in kwargs:
            self.trade_amount_usd = kwargs['trade_amount']
            if self.trading_order_manager:
                self.trading_order_manager.trade_amount_usd = kwargs['trade_amount']
            logger.info(f"Updated trade amount to ${kwargs['trade_amount']}")
        
        if 'environment' in kwargs:
            self.trading_environment = kwargs['environment']
            logger.info(f"Updated environment to {kwargs['environment']}")
            logger.warning("Restart trading flow to apply environment change")
        
        if 'symbols' in kwargs:
            self.symbols = kwargs['symbols']
            logger.info(f"Updated symbols to {kwargs['symbols']}")
            logger.warning("Restart trading flow to apply symbol changes")

    def stop_scheduler(self):
        """
        Stop the workflow scheduler and data collection
        """
        logger.info("Stopping workflow scheduler...")
        
        if self.scheduler.running:
            self.scheduler.shutdown()
        
        self.data_generator.stop_data_collection()
        self.is_running = False
        
        logger.info("Workflow scheduler stopped")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.is_running:
            self.stop_scheduler()

    # Legacy method for backward compatibility
    @staticmethod
    def training():
        """
        Legacy training method for backward compatibility
        
        Returns:
        - Trained model using default configuration
        """
        logger.warning("Using legacy training method - consider using WorkflowOrchestrator instance")
        
        orchestrator = WorkflowOrchestrator()
        
        # Try to load existing data or create mock data
        df = orchestrator.load_historical_data()
        
        if df.empty:
            logger.warning("No historical data found, creating mock data for testing")
            import pandas as pd
            import numpy as np
            
            # Create minimal mock data for testing
            mock_data = {
                'symbol': ['ETH-USD'] * 100,
                'timestamp': pd.date_range('2025-01-01', periods=100, freq='1min'),
                'mid_price': 3500 + np.random.randn(100) * 10,
                'bid_price': 3499 + np.random.randn(100) * 10,
                'ask_price': 3501 + np.random.randn(100) * 10,
                'bid_size': np.random.uniform(1, 5, 100),
                'ask_size': np.random.uniform(1, 5, 100),
                'spread_percentage': np.random.uniform(0.01, 0.1, 100),
                'volume_imbalance': np.random.uniform(-0.5, 0.5, 100)
            }
            df = pd.DataFrame(mock_data)
        
        model = orchestrator.train_model(df)
        return model
