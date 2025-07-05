#!/usr/bin/env python3
"""
Test script for WorkflowOrchestrator functionality
"""

import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
import os

from src.orchestrator.workflow_orchestrator import WorkflowOrchestrator
from src.data.historical_market_data_generator import HistoricalMarketDataGenerator
from src.model_trainer.model_trainer import ModelTrainer


class TestWorkflowOrchestrator(unittest.TestCase):
    """Test cases for WorkflowOrchestrator"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directories for testing
        self.test_data_dir = tempfile.mkdtemp(prefix="test_data_")
        self.test_model_dir = tempfile.mkdtemp(prefix="test_models_")
        
        # Initialize orchestrator with test directories
        self.orchestrator = WorkflowOrchestrator(
            data_dir=self.test_data_dir,
            model_dir=self.test_model_dir,
            symbols=['ETH-USD'],
            model_type='catboost'
        )
    
    def tearDown(self):
        """Clean up test environment"""
        # Stop any running processes
        if hasattr(self.orchestrator, 'data_generator'):
            self.orchestrator.data_generator.stop_data_collection()
        
        if hasattr(self.orchestrator, 'scheduler') and self.orchestrator.is_running:
            self.orchestrator.stop_scheduler()
        
        # Remove temporary directories
        shutil.rmtree(self.test_data_dir, ignore_errors=True)
        shutil.rmtree(self.test_model_dir, ignore_errors=True)
    
    def test_orchestrator_initialization(self):
        """Test WorkflowOrchestrator initialization"""
        self.assertIsNotNone(self.orchestrator)
        self.assertEqual(self.orchestrator.symbols, ['ETH-USD'])
        self.assertEqual(self.orchestrator.model_type, 'catboost')
        self.assertFalse(self.orchestrator.is_running)
        
        # Check that directories were created
        self.assertTrue(os.path.exists(self.test_data_dir))
        self.assertTrue(os.path.exists(self.test_model_dir))
    
    def test_status_reporting(self):
        """Test status reporting functionality"""
        status = self.orchestrator.get_status()
        
        self.assertIn('is_running', status)
        self.assertIn('symbols', status)
        self.assertIn('model_type', status)
        self.assertIn('data_collection', status)
        self.assertIn('registered_models', status)
        
        self.assertEqual(status['symbols'], ['ETH-USD'])
        self.assertEqual(status['model_type'], 'catboost')
        self.assertFalse(status['is_running'])
    
    def test_historical_data_loading_empty(self):
        """Test loading historical data when no data exists"""
        df = self.orchestrator.load_historical_data()
        self.assertTrue(df.empty)
    
    def test_training_with_mock_data(self):
        """Test model training with mock data"""
        # Create mock historical data
        mock_data = self._create_mock_data()
        
        # Test training
        model = self.orchestrator.train_model(mock_data)
        
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(hasattr(model, 'is_trained'))
    
    def test_model_evaluation(self):
        """Test model evaluation"""
        # Create mock data and train model
        mock_data = self._create_mock_data()
        model = self.orchestrator.train_model(mock_data)
        
        # Evaluate model
        metrics = self.orchestrator.evaluate_model(model, mock_data)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('rmse', metrics)
        self.assertIn('evaluation_samples', metrics)
    
    def test_model_deployment(self):
        """Test model deployment and registry"""
        # Create and train model
        mock_data = self._create_mock_data()
        model = self.orchestrator.train_model(mock_data)
        
        # Create mock metrics
        metrics = {'rmse': 0.5, 'mae': 0.3}
        
        # Deploy model
        model_path = self.orchestrator.deploy_model(model, metrics)
        
        # Check that model file was created
        self.assertTrue(os.path.exists(model_path))
        
        # Check that model was registered
        registered_models = self.orchestrator.model_registry.list_models()
        self.assertIn('catboost_latest', registered_models)
        
        # Test retrieving current model
        current_model = self.orchestrator.get_current_model()
        self.assertIsNotNone(current_model)
    
    def test_complete_training_pipeline(self):
        """Test the complete training pipeline"""
        # Create mock data file
        mock_data = self._create_mock_data()
        
        # Save mock data to simulate historical data
        filename = f"ETH-USD_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.test_data_dir, filename)
        mock_data.to_csv(filepath, index=False)
        
        # Run training pipeline
        trained_model = self.orchestrator.run_training_pipeline()
        
        self.assertIsNotNone(trained_model)
        self.assertIsNotNone(self.orchestrator.last_training)
        self.assertIsNotNone(self.orchestrator.current_model)
    
    def test_manual_triggers(self):
        """Test manual trigger methods"""
        # Test manual data save (should handle empty data gracefully)
        saved_files = self.orchestrator.manual_data_save()
        self.assertIsInstance(saved_files, dict)
        
        # Create mock data for training test
        mock_data = self._create_mock_data()
        filename = f"ETH-USD_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.test_data_dir, filename)
        mock_data.to_csv(filepath, index=False)
        
        # Test manual training
        trained_model = self.orchestrator.manual_training()
        self.assertIsNotNone(trained_model)
    
    def _create_mock_data(self):
        """Create mock historical data for testing"""
        np.random.seed(42)  # For reproducible tests
        
        n_samples = 100
        mock_data = {
            'symbol': ['ETH-USD'] * n_samples,
            'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='1min'),
            'mid_price': 3500 + np.cumsum(np.random.randn(n_samples) * 0.1),
            'bid_size': np.random.uniform(1, 5, n_samples),
            'ask_size': np.random.uniform(1, 5, n_samples),
            'spread_percentage': np.random.uniform(0.01, 0.1, n_samples),
            'volume_imbalance': np.random.uniform(-0.5, 0.5, n_samples),
            'total_volume': np.random.uniform(10, 100, n_samples),
            'bid_depth': np.random.uniform(50, 200, n_samples),
            'ask_depth': np.random.uniform(50, 200, n_samples),
            'spread_absolute': np.random.uniform(0.5, 2.0, n_samples),
            'buy_volume': np.random.uniform(5, 50, n_samples),
            'sell_volume': np.random.uniform(5, 50, n_samples)
        }
        
        df = pd.DataFrame(mock_data)
        df['bid_price'] = df['mid_price'] - 0.5
        df['ask_price'] = df['mid_price'] + 0.5
        
        return df


class TestHistoricalMarketDataGenerator(unittest.TestCase):
    """Test cases for HistoricalMarketDataGenerator"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_data_dir = tempfile.mkdtemp(prefix="test_hist_data_")
        self.generator = HistoricalMarketDataGenerator(
            data_dir=self.test_data_dir,
            retention_days=7
        )
    
    def tearDown(self):
        """Clean up test environment"""
        self.generator.stop_data_collection()
        shutil.rmtree(self.test_data_dir, ignore_errors=True)
    
    def test_generator_initialization(self):
        """Test generator initialization"""
        self.assertIsNotNone(self.generator)
        self.assertEqual(self.generator.retention_days, 7)
        self.assertFalse(self.generator.is_collecting)
        self.assertTrue(os.path.exists(self.test_data_dir))
    
    def test_data_processing(self):
        """Test market data processing"""
        # Create mock market data
        mock_market_data = {
            'symbol': 'ETH-USD',
            'timestamp': '2025-01-07T16:00:00Z',
            'type': 'l2update',
            'best_bid': {'price': '3500.0', 'size': '1.5'},
            'best_ask': {'price': '3502.0', 'size': '2.0'},
            'spread': {'absolute': 2.0, 'percentage': 0.057, 'mid_price': '3501.0'},
            'top_bids': [['3500.0', '1.5'], ['3499.0', '1.0']],
            'top_asks': [['3502.0', '2.0'], ['3503.0', '1.5']],
            'changes': [['buy', '3500.0', '1.5'], ['sell', '3502.0', '2.0']]
        }
        
        # Process the data
        processed = self.generator._process_market_data(mock_market_data)
        
        self.assertIn('symbol', processed)
        self.assertIn('bid_price', processed)
        self.assertIn('ask_price', processed)
        self.assertIn('mid_price', processed)
        self.assertEqual(processed['symbol'], 'ETH-USD')
        self.assertEqual(processed['bid_price'], 3500.0)
        self.assertEqual(processed['ask_price'], 3502.0)
    
    def test_status_reporting(self):
        """Test status reporting"""
        status = self.generator.get_collection_status()
        
        self.assertIn('is_collecting', status)
        self.assertIn('symbols', status)
        self.assertIn('buffer_sizes', status)
        self.assertIn('data_dir', status)
        self.assertIn('retention_days', status)
        
        self.assertFalse(status['is_collecting'])
        self.assertEqual(status['retention_days'], 7)


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer"""
    
    def setUp(self):
        """Set up test environment"""
        self.trainer = ModelTrainer(model_type='catboost')
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        self.assertIsNotNone(self.trainer)
        self.assertEqual(self.trainer.model_type, 'catboost')
        self.assertIsNone(self.trainer.model)
    
    def test_feature_selection(self):
        """Test feature selection"""
        # Create mock data
        mock_data = pd.DataFrame({
            'symbol': ['ETH-USD'] * 10,
            'timestamp': pd.date_range('2025-01-01', periods=10, freq='1min'),
            'mid_price': np.random.randn(10) + 3500,
            'bid_price': np.random.randn(10) + 3499,
            'ask_price': np.random.randn(10) + 3501,
            'spread_percentage': np.random.uniform(0.01, 0.1, 10),
            'volume_imbalance': np.random.uniform(-0.5, 0.5, 10),
            'irrelevant_column': np.random.randn(10)
        })
        
        features = self.trainer._select_features(mock_data)
        
        self.assertIsInstance(features, list)
        self.assertIn('mid_price', features)
        self.assertIn('bid_price', features)
        self.assertIn('spread_percentage', features)
        self.assertNotIn('symbol', features)
        self.assertNotIn('timestamp', features)
    
    def test_training_summary(self):
        """Test training summary"""
        summary = self.trainer.get_training_summary()
        
        self.assertIn('model_type', summary)
        self.assertIn('is_trained', summary)
        self.assertEqual(summary['model_type'], 'catboost')
        self.assertFalse(summary['is_trained'])


def run_tests():
    """Run all tests"""
    print("🧪 Running WorkflowOrchestrator Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestWorkflowOrchestrator))
    test_suite.addTest(unittest.makeSuite(TestHistoricalMarketDataGenerator))
    test_suite.addTest(unittest.makeSuite(TestModelTrainer))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Tests Run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
