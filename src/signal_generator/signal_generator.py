from typing import List, Dict, Optional, TYPE_CHECKING
import logging
from src.models.target_model import TargetModel
from src.market_feed.market_feed import MarketDataFeed

if TYPE_CHECKING:
    from src.order_manager.order_manager import OrderManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalGenerator:
    def __init__(self, target_model: TargetModel, market_data_feed: MarketDataFeed, subscribers: List['OrderManager'] = None):
        """
        Initialize the SignalGenerator with a target model and market data feed.

        Parameters:
        - target_model: An instance of TargetModel for generating trading signals
        - market_data_feed: An instance of MarketDataFeed for receiving market data
        - subscribers: List of OrderManager instances to receive signals

        Returns:
        - None
        """
        self._subscribers = subscribers or []
        self._target_model = target_model
        self._market_data_feed = market_data_feed
        self.signals = []
        self.latest_market_data = None
        
        # Market analysis state
        self.price_history = []
        self.order_flow_history = []
        self.spread_history = []
        
        logger.info("SignalGenerator initialized with Level 2 market data support")
        
    def start_listening(self, symbol: str = 'ETH-USD'):
        """
        Start listening to market data for signal generation
        
        Parameters:
        - symbol: Trading pair symbol to monitor
        """
        logger.info(f"Starting market data listener for {symbol}")
        self._market_data_feed.listen_to_data(symbol, self._market_data_callback)
        
    def _market_data_callback(self, market_data: Dict):
        """
        Callback function to receive market data from MarketDataFeed
        This is called automatically when new Level 2 data arrives
        
        Parameters:
        - market_data: Level 2 order book data with delta changes and best bid/ask
        """
        try:
            # Store latest market data
            self.latest_market_data = market_data
            
            # Process the market data and generate signals
            self.refresh(market_data)
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
        
    def refresh(self, market_data: Dict):
        """
        Refresh the signal generator's state with new Level 2 market data
        
        Parameters:
        - market_data: Level 2 order book data containing:
          - symbol: Trading pair
          - timestamp: Update time
          - type: 'snapshot' or 'l2update'
          - changes: Delta changes since last update
          - best_bid/best_ask: Current market
          - spread: Spread information
          - top_bids/top_asks: Order book depth
        """
        try:
            # Extract key market metrics
            market_features = self._extract_market_features(market_data)
            
            # Generate signals using the target model
            self.signals = self.generate_signals(market_features)
            
            # Publish signals to subscribers
            self.publish_signals()
            
            # Update historical data for trend analysis
            self._update_history(market_data)
            
            logger.debug(f"Generated {len(self.signals)} signals from {market_data['symbol']} data")
            
        except Exception as e:
            logger.error(f"Error in refresh: {e}")
    
    def _extract_market_features(self, market_data: Dict) -> Dict:
        """
        Extract trading features from Level 2 market data
        
        Parameters:
        - market_data: Raw Level 2 order book data
        
        Returns:
        - Dictionary of features for the ML model
        """
        features = {
            'symbol': market_data['symbol'],
            'timestamp': market_data['timestamp'],
            'update_type': market_data['type']
        }
        
        # Price features
        if market_data['best_bid'] and market_data['best_ask']:
            bid_price = float(market_data['best_bid']['price'])
            ask_price = float(market_data['best_ask']['price'])
            mid_price = (bid_price + ask_price) / 2
            
            features.update({
                'bid_price': bid_price,
                'ask_price': ask_price,
                'mid_price': mid_price,
                'bid_size': float(market_data['best_bid']['size']),
                'ask_size': float(market_data['best_ask']['size'])
            })
        
        # Spread features
        if market_data['spread']:
            features.update({
                'spread_absolute': float(market_data['spread']['absolute']),
                'spread_percentage': market_data['spread']['percentage'],
                'spread_mid_price': float(market_data['spread']['mid_price'])
            })
        
        # Order flow features from delta changes
        order_flow = self._analyze_order_flow(market_data['changes'])
        features.update(order_flow)
        
        # Order book depth features
        depth_features = self._analyze_order_book_depth(
            market_data['top_bids'], 
            market_data['top_asks']
        )
        features.update(depth_features)
        
        return features
    
    def _analyze_order_flow(self, changes: List[List[str]]) -> Dict:
        """
        Analyze order flow from delta changes
        
        Parameters:
        - changes: List of [side, price, size] changes
        
        Returns:
        - Dictionary of order flow metrics
        """
        buy_volume = 0
        sell_volume = 0
        buy_orders = 0
        sell_orders = 0
        
        for side, price, size in changes:
            size_float = float(size)
            price_float = float(price)
            
            if side == 'buy':
                if size_float > 0:  # New or increased order
                    buy_volume += size_float
                    buy_orders += 1
            elif side == 'sell':
                if size_float > 0:  # New or increased order
                    sell_volume += size_float
                    sell_orders += 1
        
        total_volume = buy_volume + sell_volume
        
        return {
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'total_volume': total_volume,
            'buy_orders': buy_orders,
            'sell_orders': sell_orders,
            'volume_imbalance': (buy_volume - sell_volume) / max(total_volume, 1),
            'order_imbalance': (buy_orders - sell_orders) / max(buy_orders + sell_orders, 1)
        }
    
    def _analyze_order_book_depth(self, bids: List[List[str]], asks: List[List[str]]) -> Dict:
        """
        Analyze order book depth and liquidity
        
        Parameters:
        - bids: List of [price, size] bid levels
        - asks: List of [price, size] ask levels
        
        Returns:
        - Dictionary of depth metrics
        """
        if not bids or not asks:
            return {
                'bid_depth': 0,
                'ask_depth': 0,
                'total_depth': 0,
                'depth_imbalance': 0,
                'weighted_mid_price': 0
            }
        
        # Calculate total depth
        bid_depth = sum(float(size) for price, size in bids)
        ask_depth = sum(float(size) for price, size in asks)
        total_depth = bid_depth + ask_depth
        
        # Calculate weighted mid price (volume-weighted)
        if bid_depth > 0 and ask_depth > 0:
            weighted_bid = sum(float(price) * float(size) for price, size in bids) / bid_depth
            weighted_ask = sum(float(price) * float(size) for price, size in asks) / ask_depth
            weighted_mid_price = (weighted_bid + weighted_ask) / 2
        else:
            weighted_mid_price = 0
        
        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'total_depth': total_depth,
            'depth_imbalance': (bid_depth - ask_depth) / max(total_depth, 1),
            'weighted_mid_price': weighted_mid_price,
            'bid_levels': len(bids),
            'ask_levels': len(asks)
        }
    
    def _update_history(self, market_data: Dict):
        """
        Update historical data for trend analysis
        
        Parameters:
        - market_data: Current market data
        """
        max_history = 100  # Keep last 100 updates
        
        # Update price history
        if market_data['best_bid'] and market_data['best_ask']:
            mid_price = (
                float(market_data['best_bid']['price']) + 
                float(market_data['best_ask']['price'])
            ) / 2
            self.price_history.append(mid_price)
            if len(self.price_history) > max_history:
                self.price_history.pop(0)
        
        # Update spread history
        if market_data['spread']:
            self.spread_history.append(market_data['spread']['percentage'])
            if len(self.spread_history) > max_history:
                self.spread_history.pop(0)
    
    def publish_signals(self):
        """
        Publish trading signals to all subscribed order managers.
        """
        if not self.signals:
            return
            
        logger.info(f"Publishing {len(self.signals)} signals to {len(self._subscribers)} subscribers")
        
        for subscriber in self._subscribers:
            try:
                subscriber.refresh(self.signals)
            except Exception as e:
                logger.error(f"Error publishing signals to subscriber: {e}")
    
    def generate_signals(self, market_features: Dict) -> List[Dict]:
        """
        Generate trading signals based on market features using the target model.

        Parameters:
        - market_features: Processed market data features

        Returns:
        - A list of trading signals generated by the target model.
        """
        try:
            # Use the target model to predict signals based on market features
            signals = self._target_model.predict(market_features)
            
            # Ensure signals is a list
            if not isinstance(signals, list):
                signals = [signals] if signals else []
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    def get_market_summary(self) -> Optional[Dict]:
        """
        Get current market summary
        
        Returns:
        - Dictionary with current market state summary
        """
        if not self.latest_market_data:
            return None
            
        return {
            'symbol': self.latest_market_data['symbol'],
            'best_bid': self.latest_market_data['best_bid'],
            'best_ask': self.latest_market_data['best_ask'],
            'spread': self.latest_market_data['spread'],
            'last_update': self.latest_market_data['timestamp'],
            'price_trend': self._calculate_price_trend(),
            'recent_signals': len(self.signals)
        }
    
    def _calculate_price_trend(self) -> str:
        """Calculate recent price trend from history"""
        if len(self.price_history) < 5:
            return 'insufficient_data'
            
        recent_prices = self.price_history[-5:]
        if recent_prices[-1] > recent_prices[0]:
            return 'upward'
        elif recent_prices[-1] < recent_prices[0]:
            return 'downward'
        else:
            return 'sideways'
