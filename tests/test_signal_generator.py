#!/usr/bin/env python3
"""
Comprehensive test suite for SignalGenerator
Tests all functionality including market data processing, signal generation, and error handling
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os
import time
from typing import Dict, List

# Add src directory to path
sys.path.append('src')

from signal_generator.signal_generator import SignalGenerator
from models.target_model import TargetModel
from market_feed.market_feed import MarketDataFeed
from order_manager.order_manager import OrderManager


class MockTargetModel(TargetModel):
    """Mock implementation of TargetModel for testing"""
    
    def __init__(self, return_signals=None):
        self.return_signals = return_signals or [{'action': 'buy', 'confidence': 0.8, 'price': 2500.0}]
        self.predict_calls = []
    
    def predict(self, data):
        self.predict_calls.append(data)
        return self.return_signals
    
    def train(self, data, target):
        pass
    
    def evaluate(self, data, target):
        return {'accuracy': 0.95}
    
    def save_model(self, filepath):
        pass


class TestSignalGeneratorInit(unittest.TestCase):
    """Test SignalGenerator initialization"""
    
    def setUp(self):
        self.mock_target_model = MockTargetModel()
        self.mock_market_feed = Mock(spec=MarketDataFeed)
        self.mock_order_manager = Mock(spec=OrderManager)
    
    def test_init_with_all_parameters(self):
        """Test initialization with all parameters provided"""
        subscribers = [self.mock_order_manager]
        
        signal_gen = SignalGenerator(
            target_model=self.mock_target_model,
            market_data_feed=self.mock_market_feed,
            subscribers=subscribers
        )
        
        self.assertEqual(signal_gen._target_model, self.mock_target_model)
        self.assertEqual(signal_gen._market_data_feed, self.mock_market_feed)
        self.assertEqual(signal_gen._subscribers, subscribers)
        self.assertEqual(signal_gen.signals, [])
        self.assertIsNone(signal_gen.latest_market_data)
        self.assertEqual(signal_gen.price_history, [])
        self.assertEqual(signal_gen.order_flow_history, [])
        self.assertEqual(signal_gen.spread_history, [])
    
    def test_init_with_none_subscribers(self):
        """Test initialization with None subscribers defaults to empty list"""
        signal_gen = SignalGenerator(
            target_model=self.mock_target_model,
            market_data_feed=self.mock_market_feed,
            subscribers=None
        )
        
        self.assertEqual(signal_gen._subscribers, [])
    
    def test_init_without_subscribers(self):
        """Test initialization without subscribers parameter"""
        signal_gen = SignalGenerator(
            target_model=self.mock_target_model,
            market_data_feed=self.mock_market_feed
        )
        
        self.assertEqual(signal_gen._subscribers, [])


class TestMarketDataProcessing(unittest.TestCase):
    """Test market data processing functionality"""
    
    def setUp(self):
        self.mock_target_model = MockTargetModel()
        self.mock_market_feed = Mock(spec=MarketDataFeed)
        self.mock_order_manager = Mock(spec=OrderManager)
        
        self.signal_gen = SignalGenerator(
            target_model=self.mock_target_model,
            market_data_feed=self.mock_market_feed,
            subscribers=[self.mock_order_manager]
        )
        
        # Sample market data matching MarketDataFeed format
        self.sample_market_data = {
            'symbol': 'ETH-USD',
            'timestamp': '2025-01-05T20:18:00.000Z',
            'type': 'l2update',
            'changes': [
                ['buy', '2500.00', '1.5'],
                ['sell', '2501.00', '2.0'],
                ['buy', '2499.50', '0']  # Remove order
            ],
            'best_bid': {'price': '2500.00', 'size': '1.5'},
            'best_ask': {'price': '2501.00', 'size': '2.0'},
            'spread': {
                'absolute': '1.00',
                'percentage': 0.04,
                'mid_price': '2500.50'
            },
            'top_bids': [
                ['2500.00', '1.5'],
                ['2499.50', '3.0'],
                ['2499.00', '2.5']
            ],
            'top_asks': [
                ['2501.00', '2.0'],
                ['2501.50', '1.8'],
                ['2502.00', '3.2']
            ]
        }
    
    def test_market_data_callback_success(self):
        """Test successful market data callback processing"""
        with patch.object(self.signal_gen, 'refresh') as mock_refresh:
            self.signal_gen._market_data_callback(self.sample_market_data)
            
            self.assertEqual(self.signal_gen.latest_market_data, self.sample_market_data)
            mock_refresh.assert_called_once_with(self.sample_market_data)
    
    def test_market_data_callback_error_handling(self):
        """Test error handling in market data callback"""
        with patch.object(self.signal_gen, 'refresh', side_effect=Exception("Test error")):
            # Should not raise exception, just log error
            self.signal_gen._market_data_callback(self.sample_market_data)
            self.assertEqual(self.signal_gen.latest_market_data, self.sample_market_data)
    
    def test_refresh_complete_flow(self):
        """Test complete refresh flow with market data"""
        with patch.object(self.signal_gen, 'publish_signals') as mock_publish:
            self.signal_gen.refresh(self.sample_market_data)
            
            # Check that signals were generated
            self.assertTrue(len(self.signal_gen.signals) > 0)
            
            # Check that publish_signals was called
            mock_publish.assert_called_once()
            
            # Check that history was updated
            self.assertTrue(len(self.signal_gen.price_history) > 0)
            self.assertTrue(len(self.signal_gen.spread_history) > 0)
    
    def test_refresh_error_handling(self):
        """Test error handling in refresh method"""
        # Create malformed market data
        bad_data = {'symbol': 'ETH-USD'}  # Missing required fields
        
        # Should not raise exception
        self.signal_gen.refresh(bad_data)
        
        # Signals should remain empty due to error
        self.assertEqual(self.signal_gen.signals, [])


class TestFeatureExtraction(unittest.TestCase):
    """Test market feature extraction methods"""
    
    def setUp(self):
        self.mock_target_model = MockTargetModel()
        self.mock_market_feed = Mock(spec=MarketDataFeed)
        
        self.signal_gen = SignalGenerator(
            target_model=self.mock_target_model,
            market_data_feed=self.mock_market_feed
        )
        
        self.sample_market_data = {
            'symbol': 'ETH-USD',
            'timestamp': '2025-01-05T20:18:00.000Z',
            'type': 'l2update',
            'changes': [
                ['buy', '2500.00', '1.5'],
                ['sell', '2501.00', '2.0'],
                ['buy', '2499.50', '0']
            ],
            'best_bid': {'price': '2500.00', 'size': '1.5'},
            'best_ask': {'price': '2501.00', 'size': '2.0'},
            'spread': {
                'absolute': '1.00',
                'percentage': 0.04,
                'mid_price': '2500.50'
            },
            'top_bids': [
                ['2500.00', '1.5'],
                ['2499.50', '3.0']
            ],
            'top_asks': [
                ['2501.00', '2.0'],
                ['2501.50', '1.8']
            ]
        }
    
    def test_extract_market_features_complete(self):
        """Test complete market feature extraction"""
        features = self.signal_gen._extract_market_features(self.sample_market_data)
        
        # Basic features
        self.assertEqual(features['symbol'], 'ETH-USD')
        self.assertEqual(features['timestamp'], '2025-01-05T20:18:00.000Z')
        self.assertEqual(features['update_type'], 'l2update')
        
        # Price features
        self.assertEqual(features['bid_price'], 2500.00)
        self.assertEqual(features['ask_price'], 2501.00)
        self.assertEqual(features['mid_price'], 2500.50)
        self.assertEqual(features['bid_size'], 1.5)
        self.assertEqual(features['ask_size'], 2.0)
        
        # Spread features
        self.assertEqual(features['spread_absolute'], 1.00)
        self.assertEqual(features['spread_percentage'], 0.04)
        self.assertEqual(features['spread_mid_price'], 2500.50)
        
        # Order flow features should be present
        self.assertIn('buy_volume', features)
        self.assertIn('sell_volume', features)
        self.assertIn('volume_imbalance', features)
        
        # Depth features should be present
        self.assertIn('bid_depth', features)
        self.assertIn('ask_depth', features)
        self.assertIn('depth_imbalance', features)
    
    def test_analyze_order_flow(self):
        """Test order flow analysis"""
        changes = [
            ['buy', '2500.00', '1.5'],   # Buy order
            ['sell', '2501.00', '2.0'],  # Sell order
            ['buy', '2499.50', '0'],     # Remove buy order
            ['sell', '2502.00', '1.0']   # Another sell order
        ]
        
        flow_metrics = self.signal_gen._analyze_order_flow(changes)
        
        self.assertEqual(flow_metrics['buy_volume'], 1.5)
        self.assertEqual(flow_metrics['sell_volume'], 3.0)  # 2.0 + 1.0
        self.assertEqual(flow_metrics['total_volume'], 4.5)
        self.assertEqual(flow_metrics['buy_orders'], 1)
        self.assertEqual(flow_metrics['sell_orders'], 2)
        
        # Volume imbalance: (1.5 - 3.0) / 4.5 = -0.333...
        self.assertAlmostEqual(flow_metrics['volume_imbalance'], -0.333, places=2)
        
        # Order imbalance: (1 - 2) / 3 = -0.333...
        self.assertAlmostEqual(flow_metrics['order_imbalance'], -0.333, places=2)
    
    def test_analyze_order_book_depth(self):
        """Test order book depth analysis"""
        bids = [['2500.00', '1.5'], ['2499.50', '3.0']]
        asks = [['2501.00', '2.0'], ['2501.50', '1.8']]
        
        depth_metrics = self.signal_gen._analyze_order_book_depth(bids, asks)
        
        self.assertEqual(depth_metrics['bid_depth'], 4.5)  # 1.5 + 3.0
        self.assertEqual(depth_metrics['ask_depth'], 3.8)  # 2.0 + 1.8
        self.assertEqual(depth_metrics['total_depth'], 8.3)
        self.assertEqual(depth_metrics['bid_levels'], 2)
        self.assertEqual(depth_metrics['ask_levels'], 2)
        
        # Depth imbalance: (4.5 - 3.8) / 8.3
        expected_imbalance = (4.5 - 3.8) / 8.3
        self.assertAlmostEqual(depth_metrics['depth_imbalance'], expected_imbalance, places=3)
    
    def test_analyze_order_book_depth_empty(self):
        """Test order book depth analysis with empty data"""
        depth_metrics = self.signal_gen._analyze_order_book_depth([], [])
        
        self.assertEqual(depth_metrics['bid_depth'], 0)
        self.assertEqual(depth_metrics['ask_depth'], 0)
        self.assertEqual(depth_metrics['total_depth'], 0)
        self.assertEqual(depth_metrics['depth_imbalance'], 0)
        self.assertEqual(depth_metrics['weighted_mid_price'], 0)


class TestSignalGeneration(unittest.TestCase):
    """Test signal generation functionality"""
    
    def setUp(self):
        self.mock_target_model = MockTargetModel()
        self.mock_market_feed = Mock(spec=MarketDataFeed)
        
        self.signal_gen = SignalGenerator(
            target_model=self.mock_target_model,
            market_data_feed=self.mock_market_feed
        )
    
    def test_generate_signals_success(self):
        """Test successful signal generation"""
        test_features = {
            'bid_price': 2500.0,
            'ask_price': 2501.0,
            'volume_imbalance': 0.2
        }
        
        expected_signals = [
            {'action': 'buy', 'confidence': 0.8, 'price': 2500.0},
            {'action': 'sell', 'confidence': 0.6, 'price': 2501.0}
        ]
        
        self.mock_target_model.return_signals = expected_signals
        
        signals = self.signal_gen.generate_signals(test_features)
        
        self.assertEqual(signals, expected_signals)
        self.assertEqual(len(self.mock_target_model.predict_calls), 1)
        self.assertEqual(self.mock_target_model.predict_calls[0], test_features)
    
    def test_generate_signals_single_signal(self):
        """Test signal generation returning single signal (not list)"""
        test_features = {'bid_price': 2500.0}
        single_signal = {'action': 'hold', 'confidence': 0.5}
        
        self.mock_target_model.return_signals = single_signal
        
        signals = self.signal_gen.generate_signals(test_features)
        
        self.assertEqual(signals, [single_signal])
    
    def test_generate_signals_none_result(self):
        """Test signal generation returning None"""
        test_features = {'bid_price': 2500.0}
        
        self.mock_target_model.return_signals = None
        
        signals = self.signal_gen.generate_signals(test_features)
        
        self.assertEqual(signals, [])
    
    def test_generate_signals_error_handling(self):
        """Test error handling in signal generation"""
        test_features = {'bid_price': 2500.0}
        
        # Mock target model to raise exception
        self.mock_target_model.predict = Mock(side_effect=Exception("Model error"))
        
        signals = self.signal_gen.generate_signals(test_features)
        
        self.assertEqual(signals, [])


class TestPublisherSubscriber(unittest.TestCase):
    """Test publisher-subscriber functionality"""
    
    def setUp(self):
        self.mock_target_model = MockTargetModel()
        self.mock_market_feed = Mock(spec=MarketDataFeed)
        self.mock_subscriber1 = Mock(spec=OrderManager)
        self.mock_subscriber2 = Mock(spec=OrderManager)
        
        self.signal_gen = SignalGenerator(
            target_model=self.mock_target_model,
            market_data_feed=self.mock_market_feed,
            subscribers=[self.mock_subscriber1, self.mock_subscriber2]
        )
    
    def test_publish_signals_success(self):
        """Test successful signal publishing to all subscribers"""
        test_signals = [
            {'action': 'buy', 'confidence': 0.8},
            {'action': 'sell', 'confidence': 0.6}
        ]
        
        self.signal_gen.signals = test_signals
        self.signal_gen.publish_signals()
        
        # Both subscribers should receive the signals
        self.mock_subscriber1.refresh.assert_called_once_with(test_signals)
        self.mock_subscriber2.refresh.assert_called_once_with(test_signals)
    
    def test_publish_signals_no_signals(self):
        """Test publishing when no signals are available"""
        self.signal_gen.signals = []
        self.signal_gen.publish_signals()
        
        # No subscribers should be called
        self.mock_subscriber1.refresh.assert_not_called()
        self.mock_subscriber2.refresh.assert_not_called()
    
    def test_publish_signals_subscriber_error(self):
        """Test error handling when subscriber fails"""
        test_signals = [{'action': 'buy', 'confidence': 0.8}]
        self.signal_gen.signals = test_signals
        
        # Make first subscriber raise exception
        self.mock_subscriber1.refresh.side_effect = Exception("Subscriber error")
        
        # Should not raise exception, should continue to other subscribers
        self.signal_gen.publish_signals()
        
        self.mock_subscriber1.refresh.assert_called_once_with(test_signals)
        self.mock_subscriber2.refresh.assert_called_once_with(test_signals)
    
    def test_publish_signals_no_subscribers(self):
        """Test publishing with no subscribers"""
        signal_gen = SignalGenerator(
            target_model=self.mock_target_model,
            market_data_feed=self.mock_market_feed,
            subscribers=[]
        )
        
        signal_gen.signals = [{'action': 'buy', 'confidence': 0.8}]
        
        # Should not raise exception
        signal_gen.publish_signals()


class TestHistoricalData(unittest.TestCase):
    """Test historical data management"""
    
    def setUp(self):
        self.mock_target_model = MockTargetModel()
        self.mock_market_feed = Mock(spec=MarketDataFeed)
        
        self.signal_gen = SignalGenerator(
            target_model=self.mock_target_model,
            market_data_feed=self.mock_market_feed
        )
    
    def test_update_history_price_tracking(self):
        """Test price history tracking"""
        market_data = {
            'best_bid': {'price': '2500.00'},
            'best_ask': {'price': '2502.00'},
            'spread': {'percentage': 0.08}
        }
        
        self.signal_gen._update_history(market_data)
        
        self.assertEqual(len(self.signal_gen.price_history), 1)
        self.assertEqual(self.signal_gen.price_history[0], 2501.0)  # Mid price
        self.assertEqual(len(self.signal_gen.spread_history), 1)
        self.assertEqual(self.signal_gen.spread_history[0], 0.08)
    
    def test_update_history_max_length(self):
        """Test history maintains maximum length"""
        market_data = {
            'best_bid': {'price': '2500.00'},
            'best_ask': {'price': '2502.00'},
            'spread': {'percentage': 0.08}
        }
        
        # Add 105 entries (more than max of 100)
        for i in range(105):
            market_data['best_bid']['price'] = f"{2500 + i}.00"
            market_data['best_ask']['price'] = f"{2502 + i}.00"
            self.signal_gen._update_history(market_data)
        
        # Should maintain only 100 entries
        self.assertEqual(len(self.signal_gen.price_history), 100)
        self.assertEqual(len(self.signal_gen.spread_history), 100)
        
        # Should have the most recent entries
        self.assertEqual(self.signal_gen.price_history[-1], 2605.0)  # (2604 + 2606) / 2
    
    def test_calculate_price_trend_upward(self):
        """Test upward price trend calculation"""
        # Add price history with upward trend
        self.signal_gen.price_history = [2500, 2501, 2502, 2503, 2505]
        
        trend = self.signal_gen._calculate_price_trend()
        self.assertEqual(trend, 'upward')
    
    def test_calculate_price_trend_downward(self):
        """Test downward price trend calculation"""
        # Add price history with downward trend
        self.signal_gen.price_history = [2505, 2503, 2502, 2501, 2500]
        
        trend = self.signal_gen._calculate_price_trend()
        self.assertEqual(trend, 'downward')
    
    def test_calculate_price_trend_sideways(self):
        """Test sideways price trend calculation"""
        # Add price history with no clear trend
        self.signal_gen.price_history = [2500, 2501, 2500, 2501, 2500]
        
        trend = self.signal_gen._calculate_price_trend()
        self.assertEqual(trend, 'sideways')
    
    def test_calculate_price_trend_insufficient_data(self):
        """Test price trend with insufficient data"""
        self.signal_gen.price_history = [2500, 2501]  # Less than 5 entries
        
        trend = self.signal_gen._calculate_price_trend()
        self.assertEqual(trend, 'insufficient_data')


class TestMarketSummary(unittest.TestCase):
    """Test market summary functionality"""
    
    def setUp(self):
        self.mock_target_model = MockTargetModel()
        self.mock_market_feed = Mock(spec=MarketDataFeed)
        
        self.signal_gen = SignalGenerator(
            target_model=self.mock_target_model,
            market_data_feed=self.mock_market_feed
        )
    
    def test_get_market_summary_with_data(self):
        """Test market summary with available data"""
        self.signal_gen.latest_market_data = {
            'symbol': 'ETH-USD',
            'timestamp': '2025-01-05T20:18:00.000Z',
            'best_bid': {'price': '2500.00', 'size': '1.5'},
            'best_ask': {'price': '2501.00', 'size': '2.0'},
            'spread': {'absolute': '1.00', 'percentage': 0.04}
        }
        
        self.signal_gen.signals = [{'action': 'buy'}, {'action': 'sell'}]
        self.signal_gen.price_history = [2500, 2501, 2502, 2503, 2505]
        
        summary = self.signal_gen.get_market_summary()
        
        self.assertIsNotNone(summary)
        self.assertEqual(summary['symbol'], 'ETH-USD')
        self.assertEqual(summary['best_bid'], {'price': '2500.00', 'size': '1.5'})
        self.assertEqual(summary['best_ask'], {'price': '2501.00', 'size': '2.0'})
        self.assertEqual(summary['last_update'], '2025-01-05T20:18:00.000Z')
        self.assertEqual(summary['price_trend'], 'upward')
        self.assertEqual(summary['recent_signals'], 2)
    
    def test_get_market_summary_no_data(self):
        """Test market summary with no data"""
        summary = self.signal_gen.get_market_summary()
        self.assertIsNone(summary)


class TestIntegration(unittest.TestCase):
    """Test integration functionality"""
    
    def setUp(self):
        self.mock_target_model = MockTargetModel()
        self.mock_market_feed = Mock(spec=MarketDataFeed)
        self.mock_subscriber = Mock(spec=OrderManager)
        
        self.signal_gen = SignalGenerator(
            target_model=self.mock_target_model,
            market_data_feed=self.mock_market_feed,
            subscribers=[self.mock_subscriber]
        )
    
    def test_start_listening(self):
        """Test start_listening sets up market data feed correctly"""
        symbol = 'BTC-USD'
        
        self.signal_gen.start_listening(symbol)
        
        # Should call listen_to_data with symbol and callback
        self.mock_market_feed.listen_to_data.assert_called_once()
        call_args = self.mock_market_feed.listen_to_data.call_args
        
        self.assertEqual(call_args[0][0], symbol)  # First argument should be symbol
        self.assertEqual(call_args[0][1], self.signal_gen._market_data_callback)  # Second should be callback
    
    def test_start_listening_default_symbol(self):
        """Test start_listening with default symbol"""
        self.signal_gen.start_listening()
        
        call_args = self.mock_market_feed.listen_to_data.call_args
        self.assertEqual(call_args[0][0], 'ETH-USD')  # Default symbol
    
    def test_end_to_end_flow(self):
        """Test complete end-to-end flow from market data to signal publication"""
        # Setup market data
        market_data = {
            'symbol': 'ETH-USD',
            'timestamp': '2025-01-05T20:18:00.000Z',
            'type': 'l2update',
            'changes': [['buy', '2500.00', '1.5']],
            'best_bid': {'price': '2500.00', 'size': '1.5'},
            'best_ask': {'price': '2501.00', 'size': '2.0'},
            'spread': {'absolute': '1.00', 'percentage': 0.04, 'mid_price': '2500.50'},
            'top_bids': [['2500.00', '1.5']],
            'top_asks': [['2501.00', '2.0']]
        }
        
        # Setup expected signals
        expected_signals = [{'action': 'buy', 'confidence': 0.8, 'price': 2500.0}]
        self.mock_target_model.return_signals = expected_signals
        
        # Trigger the flow
        self.signal_gen._market_data_callback(market_data)
        
        # Verify complete flow
        self.assertEqual(self.signal_gen.latest_market_data, market_data)
        self.assertEqual(self.signal_gen.signals, expected_signals)
        self.mock_subscriber.refresh.assert_called_once_with(expected_signals)
        
        # Verify target model was called with extracted features
        self.assertEqual(len(self.mock_target_model.predict_calls), 1)
        features = self.mock_target_model.predict_calls[0]
        
        self.assertEqual(features['symbol'], 'ETH-USD')
        self.assertEqual(features['bid_price'], 2500.0)
        self.assertEqual(features['ask_price'], 2501.0)
        self.assertIn('volume_imbalance', features)
        self.assertIn('depth_imbalance', features)


def run_tests():
    """Run all tests and return results"""
    # Create test suite
    test_classes = [
        TestSignalGeneratorInit,
        TestMarketDataProcessing,
        TestFeatureExtraction,
        TestSignalGeneration,
        TestPublisherSubscriber,
        TestHistoricalData,
        TestMarketSummary,
        TestIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("🧪 Running SignalGenerator Test Suite")
    print("=" * 60)
    
    result = run_tests()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"✅ Tests Run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n⚠️  Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n🎉 All tests passed! SignalGenerator is working correctly.")
    else:
        print("\n🔧 Some tests failed. Please review the issues above.")
    
    print("=" * 60)
    
    # Exit with appropriate code
    exit(0 if success else 1)
