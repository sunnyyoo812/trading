from trading_client import TradingClient
from signal_generator import SignalGenerator
class OrderManager:
    def __init__(self, trading_client: TradingClient):
        """
        Initialize the OrderManager with a trading client.

        Parameters:
        - trading_client: An instance of TradingClient that will be used to execute trades.

        Returns:
        - None
        """
        self.trading_client = trading_client
        
    
    def subscribe(self, signal_generator: SignalGenerator):
        """
        Subscribe to a signal generator to receive trading signals.

        Parameters:
        - signal_generator: An instance of SignalGenerator that provides trading signals.

        Returns:
        - None
        """
        pass
    
    def refresh(self):
        """
        Refresh the order manager's state, typically to update any cached data or
        to synchronize with the latest market conditions.

        Returns:
        - None
        """
        pass
    
    def manage_portfolio(self):
        """
        Manage the portfolio by executing trades based on the current trading signals
        and market conditions.

        Returns:
        - None
        """
        pass

    def convert_signals_to_trade(self):
        """
        Convert a trading signal into an executable trade order.

        Parameters:
        - None
        Returns:
        - An order object that can be executed by the trading client.
        """
        pass
    
    def direct_client_to_trade(self):
        """
        Direct the trading client to execute the trade order generated from the
        trading signals.

        Returns:
        - None
        """
        pass
    
    
