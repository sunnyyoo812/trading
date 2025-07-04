"""
Factory for creating Coinbase REST clients
"""

import os
from typing import Optional
from coinbase.rest import RESTClient
from dotenv import load_dotenv
from .exceptions import AuthenticationError

class CoinbaseClientFactory:
    """Factory for creating Coinbase REST clients"""
    
    @staticmethod
    def create_client(api_key: Optional[str] = None, 
                     api_secret: Optional[str] = None,
                     environment: Optional[str] = None) -> RESTClient:
        """
        Create a Coinbase REST client
        
        Args:
            api_key: API key (if None, loads from environment)
            api_secret: API secret (if None, loads from environment)
            environment: Environment ('sandbox' or 'production', if None loads from environment)
            
        Returns:
            Configured RESTClient instance
            
        Raises:
            AuthenticationError: If credentials are missing or invalid
        """
        # Load environment variables if not provided
        if not api_key or not api_secret or not environment:
            load_dotenv()
        
        # Get credentials from parameters or environment
        final_api_key = api_key or os.getenv('COINBASE_API_KEY')
        final_api_secret = api_secret or os.getenv('COINBASE_API_SECRET')
        final_environment = environment or os.getenv('COINBASE_ENVIRONMENT', 'sandbox')
        
        # Validate credentials
        if not final_api_key or not final_api_secret:
            raise AuthenticationError(
                "COINBASE_API_KEY and COINBASE_API_SECRET must be provided or set in environment variables"
            )
        
        # Determine base URL
        base_url = CoinbaseClientFactory._get_base_url(final_environment)
        
        try:
            return RESTClient(
                api_key=final_api_key,
                api_secret=final_api_secret,
                base_url=base_url
            )
        except Exception as e:
            raise AuthenticationError(f"Failed to create Coinbase client: {str(e)}")
    
    @staticmethod
    def _get_base_url(environment: str) -> str:
        """Get the appropriate base URL for the environment"""
        environment = environment.lower()
        
        if environment == "production":
            return "https://api.coinbase.com"
        elif environment == "sandbox":
            return "https://api-public.sandbox.exchange.coinbase.com"
        else:
            raise ValueError(f"Invalid environment: {environment}. Must be 'sandbox' or 'production'")
    
    @staticmethod
    def create_sandbox_client(api_key: str, api_secret: str) -> RESTClient:
        """Create a sandbox client"""
        return CoinbaseClientFactory.create_client(
            api_key=api_key,
            api_secret=api_secret,
            environment="sandbox"
        )
    
    @staticmethod
    def create_production_client(api_key: str, api_secret: str) -> RESTClient:
        """Create a production client"""
        return CoinbaseClientFactory.create_client(
            api_key=api_key,
            api_secret=api_secret,
            environment="production"
        )
