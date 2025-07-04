"""
Custom exceptions for TradingClient
"""

class TradingClientError(Exception):
    """Base exception for TradingClient errors"""
    pass

class AuthenticationError(TradingClientError):
    """Raised when API authentication fails"""
    pass

class ValidationError(TradingClientError):
    """Raised when input validation fails"""
    pass

class OrderError(TradingClientError):
    """Raised when order operations fail"""
    pass

class APIError(TradingClientError):
    """Raised when API calls fail"""
    pass
