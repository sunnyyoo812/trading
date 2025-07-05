import asyncio
import json
import logging
import os
import threading
import time
import websockets
from collections import defaultdict
from decimal import Decimal
from typing import Dict, List, Callable, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderBook:
    """Maintains an order book with top N levels for bids and asks"""
    
    def __init__(self, max_levels: int = 20):
        self.max_levels = max_levels
        self.bids: Dict[str, str] = {}  # price -> size
        self.asks: Dict[str, str] = {}  # price -> size
        self.last_update_time = None
        
    def apply_snapshot(self, bids: List[List[str]], asks: List[List[str]]):
        """Apply initial snapshot data"""
        self.bids.clear()
        self.asks.clear()
        
        # Add bids (sorted descending by price)
        for price, size in bids[:self.max_levels]:
            if float(size) > 0:
                self.bids[price] = size
                
        # Add asks (sorted ascending by price)  
        for price, size in asks[:self.max_levels]:
            if float(size) > 0:
                self.asks[price] = size
                
        self.last_update_time = time.time()
        logger.info(f"Applied snapshot: {len(self.bids)} bids, {len(self.asks)} asks")
        
    def apply_changes(self, changes: List[List[str]]) -> List[List[str]]:
        """Apply delta changes and return the actual changes made"""
        applied_changes = []
        
        for side, price, size in changes:
            size_float = float(size)
            
            if side == 'buy':
                if size_float == 0:
                    # Remove bid level
                    if price in self.bids:
                        del self.bids[price]
                        applied_changes.append([side, price, size])
                else:
                    # Add or update bid level
                    self.bids[price] = size
                    applied_changes.append([side, price, size])
                    
            elif side == 'sell':
                if size_float == 0:
                    # Remove ask level
                    if price in self.asks:
                        del self.asks[price]
                        applied_changes.append([side, price, size])
                else:
                    # Add or update ask level
                    self.asks[price] = size
                    applied_changes.append([side, price, size])
        
        # Trim to max levels and sort
        self._trim_and_sort()
        self.last_update_time = time.time()
        
        return applied_changes
    
    def _trim_and_sort(self):
        """Keep only top N levels and ensure proper sorting"""
        # Sort bids descending (highest price first)
        sorted_bids = sorted(self.bids.items(), key=lambda x: float(x[0]), reverse=True)
        self.bids = dict(sorted_bids[:self.max_levels])
        
        # Sort asks ascending (lowest price first)
        sorted_asks = sorted(self.asks.items(), key=lambda x: float(x[0]))
        self.asks = dict(sorted_asks[:self.max_levels])
    
    def get_best_bid_ask(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Get current best bid and ask"""
        best_bid = None
        best_ask = None
        
        if self.bids:
            # Best bid is highest price
            best_bid_price = max(self.bids.keys(), key=float)
            best_bid = {
                'price': best_bid_price,
                'size': self.bids[best_bid_price]
            }
            
        if self.asks:
            # Best ask is lowest price
            best_ask_price = min(self.asks.keys(), key=float)
            best_ask = {
                'price': best_ask_price,
                'size': self.asks[best_ask_price]
            }
            
        return best_bid, best_ask
    
    def get_spread_info(self) -> Optional[Dict]:
        """Calculate spread information"""
        best_bid, best_ask = self.get_best_bid_ask()
        
        if not best_bid or not best_ask:
            return None
            
        bid_price = float(best_bid['price'])
        ask_price = float(best_ask['price'])
        
        spread_abs = ask_price - bid_price
        mid_price = (bid_price + ask_price) / 2
        spread_pct = (spread_abs / mid_price) * 100 if mid_price > 0 else 0
        
        return {
            'absolute': f"{spread_abs:.2f}",
            'percentage': round(spread_pct, 4),
            'mid_price': f"{mid_price:.2f}"
        }
    
    def get_top_levels(self) -> Tuple[List[List[str]], List[List[str]]]:
        """Get top N levels as lists"""
        # Bids sorted descending (highest first)
        top_bids = [[price, size] for price, size in 
                   sorted(self.bids.items(), key=lambda x: float(x[0]), reverse=True)]
        
        # Asks sorted ascending (lowest first)  
        top_asks = [[price, size] for price, size in
                   sorted(self.asks.items(), key=lambda x: float(x[0]))]
        
        return top_bids, top_asks


class MarketDataFeed:
    """Coinbase WebSocket Level 2 Order Book Feed"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, environment: str = 'production'):
        """
        Initialize MarketDataFeed with Coinbase WebSocket connection
        
        Parameters:
        - api_key: Coinbase API key (optional for public feeds)
        - api_secret: Coinbase API secret (optional for public feeds)
        - environment: 'sandbox' or 'production'
        """
        self.api_key = api_key or os.getenv('COINBASE_API_KEY')
        self.api_secret = api_secret or os.getenv('COINBASE_API_SECRET')
        self.environment = environment or os.getenv('COINBASE_ENVIRONMENT', 'sandbox')
        
        # WebSocket URLs
        self.ws_urls = {
            'sandbox': 'wss://ws-feed-public.sandbox.exchange.coinbase.com',
            'production': 'wss://ws-feed.exchange.coinbase.com'
        }
        
        self.ws_url = self.ws_urls.get(self.environment, self.ws_urls['sandbox'])
        
        # State management
        self.order_books: Dict[str, OrderBook] = {}
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.websocket = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
        # Threading
        self.event_loop = None
        self.websocket_thread = None
        
        logger.info(f"MarketDataFeed initialized for {self.environment} environment")
    
    def listen_to_data(self, symbol: str, callback: Callable):
        """
        Listen to Level 2 order book data for a specific symbol
        
        Parameters:
        - symbol: Trading pair symbol (e.g., 'ETH-USD')
        - callback: Function to call with market data updates
        """
        if not callback:
            raise ValueError("Callback function is required")
            
        # Add callback for this symbol
        self.callbacks[symbol].append(callback)
        
        # Initialize order book for this symbol
        if symbol not in self.order_books:
            self.order_books[symbol] = OrderBook(max_levels=20)
        
        # Start WebSocket connection if not already running
        if not self.running:
            self._start_websocket_connection([symbol])
        else:
            # Subscribe to additional symbol if connection already exists
            self._subscribe_to_symbol(symbol)
            
        logger.info(f"Started listening to {symbol} with callback registered")
    
    def _start_websocket_connection(self, symbols: List[str]):
        """Start WebSocket connection in a separate thread"""
        self.running = True
        self.websocket_thread = threading.Thread(
            target=self._run_websocket_loop,
            args=(symbols,),
            daemon=True
        )
        self.websocket_thread.start()
        logger.info("WebSocket connection thread started")
    
    def _run_websocket_loop(self, symbols: List[str]):
        """Run the WebSocket event loop"""
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        
        try:
            self.event_loop.run_until_complete(self._websocket_handler(symbols))
        except Exception as e:
            logger.error(f"WebSocket loop error: {e}")
        finally:
            self.event_loop.close()
    
    async def _websocket_handler(self, symbols: List[str]):
        """Handle WebSocket connection and messages"""
        while self.running and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                logger.info(f"Connecting to {self.ws_url}")
                
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    self.websocket = websocket
                    self.reconnect_attempts = 0
                    
                    # Subscribe to level2_batch channel
                    subscribe_msg = {
                        "type": "subscribe",
                        "product_ids": symbols,
                        "channels": ["level2_batch"]
                    }
                    
                    await websocket.send(json.dumps(subscribe_msg))
                    logger.info(f"Subscribed to level2_batch for {symbols}")
                    
                    # Listen for messages
                    async for message in websocket:
                        if not self.running:
                            break
                            
                        try:
                            data = json.loads(message)
                            await self._handle_message(data)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}")
                        except Exception as e:
                            logger.error(f"Message handling error: {e}")
                            
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
            
            # Reconnection logic
            if self.running:
                self.reconnect_attempts += 1
                wait_time = min(2 ** self.reconnect_attempts, 60)  # Exponential backoff
                logger.info(f"Reconnecting in {wait_time} seconds (attempt {self.reconnect_attempts})")
                await asyncio.sleep(wait_time)
    
    async def _handle_message(self, data: Dict):
        """Handle incoming WebSocket messages"""
        msg_type = data.get('type')
        product_id = data.get('product_id')
        
        if not product_id or product_id not in self.order_books:
            return
            
        order_book = self.order_books[product_id]
        
        if msg_type == 'snapshot':
            # Initial order book snapshot
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            order_book.apply_snapshot(bids, asks)
            
            # Send initial snapshot to callbacks
            await self._send_to_callbacks(product_id, {
                'symbol': product_id,
                'timestamp': data.get('time', time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')),
                'type': 'snapshot',
                'changes': [],
                **self._get_market_data_summary(order_book)
            })
            
        elif msg_type == 'l2update':
            # Delta updates
            changes = data.get('changes', [])
            if changes:
                applied_changes = order_book.apply_changes(changes)
                
                # Send update to callbacks
                await self._send_to_callbacks(product_id, {
                    'symbol': product_id,
                    'timestamp': data.get('time', time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')),
                    'type': 'l2update',
                    'changes': applied_changes,
                    **self._get_market_data_summary(order_book)
                })
    
    def _get_market_data_summary(self, order_book: OrderBook) -> Dict:
        """Get formatted market data summary"""
        best_bid, best_ask = order_book.get_best_bid_ask()
        spread_info = order_book.get_spread_info()
        top_bids, top_asks = order_book.get_top_levels()
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread_info,
            'top_bids': top_bids,
            'top_asks': top_asks
        }
    
    async def _send_to_callbacks(self, symbol: str, data: Dict):
        """Send data to all registered callbacks for a symbol"""
        callbacks = self.callbacks.get(symbol, [])
        
        for callback in callbacks:
            try:
                # Run callback in thread pool to avoid blocking
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    # Run sync callback in executor
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, callback, data)
            except Exception as e:
                logger.error(f"Callback error for {symbol}: {e}")
    
    def _subscribe_to_symbol(self, symbol: str):
        """Subscribe to additional symbol (if WebSocket already connected)"""
        # This would require sending additional subscribe messages
        # For now, we'll handle this in future iterations
        logger.info(f"Additional subscription to {symbol} requested")
    
    def publish_market_data(self):
        """
        Publish market data to subscribers (legacy method for compatibility)
        
        Note: In the WebSocket implementation, data is published automatically
        when received from Coinbase. This method is kept for interface compatibility.
        """
        if not self.running:
            logger.warning("WebSocket not running, cannot publish market data")
            return
            
        logger.info("Market data is being published automatically via WebSocket callbacks")
    
    def stop_feed(self):
        """Stop the market data feed and cleanup resources"""
        logger.info("Stopping market data feed...")
        self.running = False
        
        if self.websocket:
            # Close WebSocket connection
            if self.event_loop and not self.event_loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self.websocket.close(),
                    self.event_loop
                )
        
        if self.websocket_thread and self.websocket_thread.is_alive():
            self.websocket_thread.join(timeout=5)
            
        self.callbacks.clear()
        self.order_books.clear()
        logger.info("Market data feed stopped")
    
    def get_current_order_book(self, symbol: str) -> Optional[Dict]:
        """Get current order book state for a symbol"""
        if symbol not in self.order_books:
            return None
            
        order_book = self.order_books[symbol]
        return {
            'symbol': symbol,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'type': 'current_state',
            'changes': [],
            **self._get_market_data_summary(order_book)
        }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.running:
            self.stop_feed()
