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
    def __init__(self, target_model: TargetModel, market_data_feed: MarketDataFeed, 
                 subscribers: List['OrderManager'] = None, buy_threshold: float = 2.0, 
                 sell_threshold: float = 2.0):
        """
        Initialize the SignalGenerator with a target model and market data feed.

        Parameters:
        - target_model: An instance of TargetModel for generating trading signals
        - market_data_feed: An instance of MarketDataFeed for receiving market data
        - subscribers: List of OrderManager instances to receive signals
        - buy_threshold: Minimum predicted price change percentage to trigger buy signal (default: 2.0%)
        - sell_threshold: Minimum predicted price change percentage to trigger sell signal (default: 2.0%)

        Returns:
        - None
        """
        self._subscribers = subscribers or []
        self._target_model = target_model
        self._market_data_feed = market_data_feed
        self.signals = []
        self.latest_market_data = None
        
        # Signal generation thresholds
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
        # Market analysis state
        self.price_history = []
        self.order_flow_history = []
        self.spread_history = []
        
        logger.info(f"SignalGenerator initialized with buy_threshold={buy_threshold}%, sell_threshold={sell_threshold}%")
        
    def start_listening(self, symbol: str = 'DOGE-USD'):
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
        Generate trading signals based on model predictions of price change percentage.

        Parameters:
        - market_features: Processed market data features

        Returns:
        - A list of trading signal dictionaries with action, confidence, and metadata
        """
        try:
            # Get price change prediction from the model (percentage)
            predicted_change_pct = self._target_model.predict(market_features)
            
            # Get current price for signal metadata
            current_price = market_features.get('mid_price', 0.0)
            if current_price == 0.0:
                # Fallback to bid/ask average if mid_price not available
                bid_price = market_features.get('bid_price', 0.0)
                ask_price = market_features.get('ask_price', 0.0)
                current_price = (bid_price + ask_price) / 2 if (bid_price > 0 and ask_price > 0) else 0.0
            
            # Calculate predicted target price
            predicted_price = current_price * (1 + predicted_change_pct / 100) if current_price > 0 else 0.0
            
            # Generate signal based on thresholds
            signal = self._create_signal_from_prediction(
                predicted_change_pct=predicted_change_pct,
                current_price=current_price,
                predicted_price=predicted_price,
                market_features=market_features
            )
            
            return [signal]
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return [self._create_error_signal(str(e))]
    
    def _create_signal_from_prediction(self, predicted_change_pct: float, current_price: float, 
                                     predicted_price: float, market_features: Dict) -> Dict:
        """
        Create a trading signal based on model prediction and thresholds.
        
        Parameters:
        - predicted_change_pct: Predicted price change percentage
        - current_price: Current market price
        - predicted_price: Predicted target price
        - market_features: Market features for metadata
        
        Returns:
        - Trading signal dictionary
        """
        # Determine action based on thresholds
        if predicted_change_pct >= self.buy_threshold:
            action = 'buy'
            threshold_used = self.buy_threshold
            reason = f"Model predicts +{predicted_change_pct:.2f}% price increase (≥ {self.buy_threshold}% threshold)"
        elif predicted_change_pct <= -self.sell_threshold:
            action = 'sell'
            threshold_used = self.sell_threshold
            reason = f"Model predicts {predicted_change_pct:.2f}% price decrease (≤ -{self.sell_threshold}% threshold)"
        else:
            action = 'hold'
            threshold_used = max(self.buy_threshold, self.sell_threshold)
            reason = f"Model predicts {predicted_change_pct:.2f}% price change (within ±{threshold_used}% threshold)"
        
        # Calculate confidence based on how far prediction is from threshold
        confidence = self._calculate_confidence(predicted_change_pct, action, threshold_used)
        
        # Create signal dictionary
        signal = {
            'action': action,
            'confidence': confidence,
            'predicted_change_pct': round(predicted_change_pct, 4),
            'current_price': round(current_price, 6),
            'predicted_price': round(predicted_price, 6),
            'threshold_used': threshold_used,
            'timestamp': market_features.get('timestamp', ''),
            'symbol': market_features.get('symbol', 'UNKNOWN'),
            'reason': reason,
            'signal_strength': abs(predicted_change_pct),
            'market_context': {
                'spread_pct': market_features.get('spread_percentage', 0.0),
                'volume_imbalance': market_features.get('volume_imbalance', 0.0),
                'depth_imbalance': market_features.get('depth_imbalance', 0.0)
            }
        }
        
        logger.info(f"Generated {action} signal: {predicted_change_pct:.2f}% predicted change, "
                   f"confidence: {confidence:.2f}")
        
        return signal
    
    def _calculate_confidence(self, predicted_change_pct: float, action: str, threshold_used: float) -> float:
        """
        Calculate confidence score based on prediction strength relative to threshold.
        
        Parameters:
        - predicted_change_pct: Model's predicted price change percentage
        - action: Determined action ('buy', 'sell', 'hold')
        - threshold_used: Threshold that was applied
        
        Returns:
        - Confidence score between 0.0 and 1.0
        """
        if action == 'hold':
            # For hold signals, confidence is lower when closer to thresholds
            distance_from_threshold = min(
                abs(predicted_change_pct - self.buy_threshold),
                abs(predicted_change_pct + self.sell_threshold)
            )
            # Normalize to 0-1, with max confidence at 0% change
            confidence = max(0.1, 1.0 - (abs(predicted_change_pct) / threshold_used))
        else:
            # For buy/sell signals, confidence increases with distance from threshold
            if action == 'buy':
                excess = predicted_change_pct - self.buy_threshold
            else:  # sell
                excess = abs(predicted_change_pct) - self.sell_threshold
            
            # Scale confidence: threshold = 0.5, 2x threshold = 1.0
            confidence = min(1.0, 0.5 + (excess / threshold_used) * 0.5)
            confidence = max(0.1, confidence)  # Minimum confidence
        
        return round(confidence, 3)
    
    def _create_error_signal(self, error_message: str) -> Dict:
        """
        Create a default hold signal when there's an error.
        
        Parameters:
        - error_message: Error description
        
        Returns:
        - Default hold signal
        """
        return {
            'action': 'hold',
            'confidence': 0.0,
            'predicted_change_pct': 0.0,
            'current_price': 0.0,
            'predicted_price': 0.0,
            'threshold_used': max(self.buy_threshold, self.sell_threshold),
            'timestamp': '',
            'symbol': 'UNKNOWN',
            'reason': f'Error in signal generation: {error_message}',
            'signal_strength': 0.0,
            'market_context': {}
        }
    
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
