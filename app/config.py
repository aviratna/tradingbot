"""Configuration settings for the trading dashboard."""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # API Keys (optional - will use free tiers where possible)
    NEWS_API_KEY: Optional[str] = os.getenv("NEWS_API_KEY")
    TWITTER_BEARER_TOKEN: Optional[str] = os.getenv("TWITTER_BEARER_TOKEN")

    # API Endpoints
    COINGECKO_API_URL: str = "https://api.coingecko.com/api/v3"
    EXCHANGERATE_API_URL: str = "https://open.er-api.com/v6/latest"
    NEWS_API_URL: str = "https://newsapi.org/v2"

    # Default assets to track
    DEFAULT_STOCKS: list = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    DEFAULT_CRYPTO: list = ["bitcoin", "ethereum", "solana", "cardano"]
    DEFAULT_FOREX: list = ["EUR", "GBP", "JPY", "AUD", "CAD"]

    # Precious metals settings
    DEFAULT_METALS: list = ["XAU/USD", "XAG/USD"]
    METALS_SYMBOLS: dict = {
        "XAU/USD": {"yf_symbol": "GC=F", "name": "Gold", "pip_value": 10.0},
        "XAG/USD": {"yf_symbol": "SI=F", "name": "Silver", "pip_value": 50.0},
    }
    CORRELATED_ASSETS: list = ["DXY", "US10Y", "SPY", "VIX", "USO", "EUR/USD"]

    # Trading settings
    DEFAULT_ACCOUNT_SIZE: float = 5000.0
    DEFAULT_RISK_PERCENT: float = 2.0
    MIN_RR_RATIO: float = 1.5
    MAX_DAILY_RISK_PERCENT: float = 6.0

    # Fibonacci levels
    FIBONACCI_RETRACEMENT: list = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    FIBONACCI_EXTENSION: list = [1.0, 1.272, 1.618, 2.0, 2.618]

    # Cache settings (in seconds)
    CACHE_TTL_MARKET: int = 60  # 1 minute
    CACHE_TTL_NEWS: int = 300   # 5 minutes
    CACHE_TTL_SOCIAL: int = 120  # 2 minutes

    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"


settings = Settings()
