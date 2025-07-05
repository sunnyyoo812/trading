"""
Historical Market Data Generator for collecting and storing market data from live feeds
"""

import os
import json
import pandas as pd
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict, deque
import pytz

from src.market_feed.market_feed import MarketDataFeed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalMarketDataGenerator:
    """
    Collects real-time market data and generates historical datasets for model training
    """
    
    def __init__(self, data_dir: str = "data/historical", retention_days: int = 7):
        """
        Initialize the historical data generator
        
        Parameters:
        - data_dir: Directory to store historical data files
        - retention_days: Number of days to retain historical data
        """
        self.data_dir = data_dir
        self.retention_days = retention_days
        self.market_feed = None
        self.is_collecting = False
        
        # Data storage - using deque for efficient append/pop operations
        self.market_data_buffer = defaultdict(lambda: deque(maxlen=10000))  # Per symbol buffer
        self.aggregated_data = defaultdict(list)  # Daily aggregated data per symbol
        
        # Timezone for EST scheduling
        self.est_tz = pytz.timezone('US/Eastern')
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info(f"HistoricalMarketDataGenerator initialized with {retention_days} day retention")
    
    def start_data_collection(self, symbols: List[str] = None):
        """
        Start collecting real-time market data
        
        Parameters:
        - symbols: List of trading symbols to collect data for (default: ['DOGE-USD', 'BTC-USD'])
        """
        if symbols is None:
            symbols = ['DOGE-USD', 'BTC-USD']
        
        if self.is_collecting:
            logger.warning("Data collection already running")
            return
        
        self.is_collecting = True
        
        # Initialize market feed
        self.market_feed = MarketDataFeed()
        
        # Register callbacks for each symbol
        for symbol in symbols:
            self.market_feed.listen_to_data(symbol, self._market_data_callback)
            logger.info(f"Started collecting data for {symbol}")
        
        logger.info(f"Data collection started for symbols: {symbols}")
    
    def _market_data_callback(self, market_data: Dict):
        """
        Callback function to receive and store market data
        
        Parameters:
        - market_data: Market data from MarketDataFeed
        """
        try:
            symbol = market_data.get('symbol')
            if not symbol:
                return
            
            # Extract relevant features for training
            processed_data = self._process_market_data(market_data)
            
            # Add to buffer
            self.market_data_buffer[symbol].append(processed_data)
            
            # Log periodically (every 100 updates)
            if len(self.market_data_buffer[symbol]) % 100 == 0:
                logger.debug(f"Collected {len(self.market_data_buffer[symbol])} data points for {symbol}")
                
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    def _process_market_data(self, market_data: Dict) -> Dict:
        """
        Process raw market data into features suitable for model training
        
        Parameters:
        - market_data: Raw market data from feed
        
        Returns:
        - Dictionary of processed features
        """
        timestamp = market_data.get('timestamp', datetime.now().isoformat())
        symbol = market_data.get('symbol')
        
        # Initialize processed data
        processed = {
            'timestamp': timestamp,
            'symbol': symbol,
            'update_type': market_data.get('type', 'unknown')
        }
        
        # Price and spread features
        best_bid = market_data.get('best_bid')
        best_ask = market_data.get('best_ask')
        spread_info = market_data.get('spread')
        
        if best_bid and best_ask:
            bid_price = float(best_bid['price'])
            ask_price = float(best_ask['price'])
            mid_price = (bid_price + ask_price) / 2
            
            processed.update({
                'bid_price': bid_price,
                'ask_price': ask_price,
                'mid_price': mid_price,
                'bid_size': float(best_bid['size']),
                'ask_size': float(best_ask['size'])
            })
        
        # Spread features
        if spread_info:
            processed.update({
                'spread_absolute': float(spread_info.get('absolute', 0)),
                'spread_percentage': spread_info.get('percentage', 0),
                'spread_mid_price': float(spread_info.get('mid_price', 0))
            })
        
        # Order book depth features
        top_bids = market_data.get('top_bids', [])
        top_asks = market_data.get('top_asks', [])
        
        if top_bids and top_asks:
            # Calculate depth metrics
            bid_depth = sum(float(size) for price, size in top_bids[:5])  # Top 5 levels
            ask_depth = sum(float(size) for price, size in top_asks[:5])
            total_depth = bid_depth + ask_depth
            
            processed.update({
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total_depth': total_depth,
                'depth_imbalance': (bid_depth - ask_depth) / max(total_depth, 1),
                'bid_levels': len(top_bids),
                'ask_levels': len(top_asks)
            })
        
        # Order flow features from changes
        changes = market_data.get('changes', [])
        if changes:
            buy_volume = sum(float(size) for side, price, size in changes if side == 'buy' and float(size) > 0)
            sell_volume = sum(float(size) for side, price, size in changes if side == 'sell' and float(size) > 0)
            total_volume = buy_volume + sell_volume
            
            processed.update({
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'total_volume': total_volume,
                'volume_imbalance': (buy_volume - sell_volume) / max(total_volume, 1) if total_volume > 0 else 0,
                'order_changes': len(changes)
            })
        
        return processed
    
    def save_daily_historical_data(self, target_date: datetime = None) -> Dict[str, str]:
        """
        Save collected data as historical dataset for training
        
        Parameters:
        - target_date: Date to save data for (default: today)
        
        Returns:
        - Dictionary mapping symbols to saved file paths
        """
        if target_date is None:
            target_date = datetime.now(self.est_tz).date()
        
        saved_files = {}
        
        for symbol, data_buffer in self.market_data_buffer.items():
            if not data_buffer:
                logger.warning(f"No data collected for {symbol}")
                continue
            
            try:
                # Convert buffer to DataFrame
                df = pd.DataFrame(list(data_buffer))
                
                # Add derived features
                df = self._add_derived_features(df)
                
                # Save to file
                filename = f"{symbol}_{target_date.strftime('%Y%m%d')}.csv"
                filepath = os.path.join(self.data_dir, filename)
                
                df.to_csv(filepath, index=False)
                saved_files[symbol] = filepath
                
                # Clear buffer after saving
                data_buffer.clear()
                
                logger.info(f"Saved {len(df)} records for {symbol} to {filepath}")
                
            except Exception as e:
                logger.error(f"Error saving data for {symbol}: {e}")
        
        # Clean up old files
        self._cleanup_old_files()
        
        return saved_files
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for model training
        
        Parameters:
        - df: DataFrame with raw market data
        
        Returns:
        - DataFrame with additional derived features
        """
        if df.empty:
            return df
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Price change features
        if 'mid_price' in df.columns:
            df['price_change'] = df['mid_price'].pct_change()
            df['price_change_abs'] = df['mid_price'].diff()
            
            # Rolling features (5-minute windows)
            window_size = min(50, len(df) // 4)  # Adaptive window size
            if window_size > 1:
                df['price_volatility'] = df['price_change'].rolling(window=window_size).std()
                df['price_trend'] = df['mid_price'].rolling(window=window_size).apply(
                    lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1 if x.iloc[-1] < x.iloc[0] else 0
                )
        
        # Volume features
        if 'total_volume' in df.columns:
            window_size = min(20, len(df) // 4)
            if window_size > 1:
                df['volume_ma'] = df['total_volume'].rolling(window=window_size).mean()
                df['volume_ratio'] = df['total_volume'] / df['volume_ma']
        
        # Spread features
        if 'spread_percentage' in df.columns:
            window_size = min(30, len(df) // 4)
            if window_size > 1:
                df['spread_ma'] = df['spread_percentage'].rolling(window=window_size).mean()
                df['spread_volatility'] = df['spread_percentage'].rolling(window=window_size).std()
        
        # Fill NaN values
        df = df.fillna(method='forward').fillna(0)
        
        return df
    
    def load_historical_data(self, symbols: List[str] = None, days_back: int = None) -> pd.DataFrame:
        """
        Load historical data for model training
        
        Parameters:
        - symbols: List of symbols to load (default: all available)
        - days_back: Number of days to load (default: retention_days)
        
        Returns:
        - Combined DataFrame with historical data
        """
        if days_back is None:
            days_back = self.retention_days
        
        # Get date range
        end_date = datetime.now(self.est_tz).date()
        start_date = end_date - timedelta(days=days_back)
        
        all_data = []
        
        # Get all CSV files in data directory
        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.csv'):
                continue
            
            try:
                # Parse filename: SYMBOL_YYYYMMDD.csv
                parts = filename.replace('.csv', '').split('_')
                if len(parts) != 2:
                    continue
                
                symbol, date_str = parts
                file_date = datetime.strptime(date_str, '%Y%m%d').date()
                
                # Check if file is in date range and symbol filter
                if file_date < start_date or file_date > end_date:
                    continue
                
                if symbols and symbol not in symbols:
                    continue
                
                # Load file
                filepath = os.path.join(self.data_dir, filename)
                df = pd.read_csv(filepath)
                df['file_date'] = file_date
                all_data.append(df)
                
                logger.debug(f"Loaded {len(df)} records from {filename}")
                
            except Exception as e:
                logger.error(f"Error loading file {filename}: {e}")
        
        if not all_data:
            logger.warning("No historical data found")
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df = combined_df.sort_values(['symbol', 'timestamp'])
        
        logger.info(f"Loaded {len(combined_df)} total records for {combined_df['symbol'].nunique()} symbols")
        
        return combined_df
    
    def _cleanup_old_files(self):
        """Remove data files older than retention period"""
        cutoff_date = datetime.now(self.est_tz).date() - timedelta(days=self.retention_days)
        
        removed_count = 0
        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.csv'):
                continue
            
            try:
                # Parse date from filename
                parts = filename.replace('.csv', '').split('_')
                if len(parts) != 2:
                    continue
                
                date_str = parts[1]
                file_date = datetime.strptime(date_str, '%Y%m%d').date()
                
                if file_date < cutoff_date:
                    filepath = os.path.join(self.data_dir, filename)
                    os.remove(filepath)
                    removed_count += 1
                    logger.debug(f"Removed old file: {filename}")
                    
            except Exception as e:
                logger.error(f"Error processing file {filename} for cleanup: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old data files")
    
    def stop_data_collection(self):
        """Stop collecting market data"""
        if not self.is_collecting:
            return
        
        self.is_collecting = False
        
        if self.market_feed:
            self.market_feed.stop_feed()
            self.market_feed = None
        
        logger.info("Data collection stopped")
    
    def get_collection_status(self) -> Dict:
        """Get current data collection status"""
        status = {
            'is_collecting': self.is_collecting,
            'symbols': list(self.market_data_buffer.keys()),
            'buffer_sizes': {symbol: len(buffer) for symbol, buffer in self.market_data_buffer.items()},
            'data_dir': self.data_dir,
            'retention_days': self.retention_days
        }
        
        # Count historical files
        historical_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        status['historical_files'] = len(historical_files)
        
        return status
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.is_collecting:
            self.stop_data_collection()
