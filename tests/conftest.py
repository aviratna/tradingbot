"""Pytest configuration and shared fixtures."""
import pytest
import asyncio
import sys
import os

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    import pandas as pd
    import numpy as np

    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    return pd.Series(prices, index=dates, name='TEST')


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    import pandas as pd
    import numpy as np

    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    base = 100 + np.cumsum(np.random.randn(30))

    return pd.DataFrame({
        'Open': base - np.random.rand(30),
        'High': base + np.random.rand(30) * 2,
        'Low': base - np.random.rand(30) * 2,
        'Close': base,
        'Volume': np.random.randint(1000000, 10000000, 30)
    }, index=dates)


@pytest.fixture
def bullish_texts():
    """Sample bullish text content."""
    return [
        "Bitcoin is mooning! Very bullish on crypto!",
        "Strong breakout! Rally incoming!",
        "Buy the dip! Massive gains ahead!",
        "Bullish divergence confirmed. Long position opened.",
        "To the moon! HODL strong!"
    ]


@pytest.fixture
def bearish_texts():
    """Sample bearish text content."""
    return [
        "Market crash incoming! Sell everything!",
        "Bearish trend confirmed. Major decline expected.",
        "Panic selling across all markets.",
        "Economic collapse imminent. Dump your holdings.",
        "Capitulation phase has begun."
    ]


@pytest.fixture
def neutral_texts():
    """Sample neutral text content."""
    return [
        "Market trading sideways today.",
        "Volume remains average.",
        "No significant news to report.",
        "Traders await Fed decision.",
        "Markets consolidated near support."
    ]


@pytest.fixture
def mock_news_articles():
    """Generate mock news articles for testing."""
    from datetime import datetime

    return [
        {
            "title": "Stock Market Hits New High",
            "description": "Major indices continue their bullish run.",
            "source": "Financial Times",
            "url": "https://example.com/1",
            "published_at": datetime.now()
        },
        {
            "title": "Crypto Market Sees Increased Activity",
            "description": "Bitcoin and Ethereum lead the rally.",
            "source": "CoinDesk",
            "url": "https://example.com/2",
            "published_at": datetime.now()
        }
    ]


@pytest.fixture
def mock_social_posts():
    """Generate mock social media posts for testing."""
    from datetime import datetime

    return [
        {
            "id": "1",
            "text": "Bullish on $AAPL! Great earnings!",
            "author": "trader1",
            "created_at": datetime.now(),
            "likes": 100,
            "retweets": 20
        },
        {
            "id": "2",
            "text": "Bearish on crypto right now. Too much uncertainty.",
            "author": "trader2",
            "created_at": datetime.now(),
            "likes": 50,
            "retweets": 10
        }
    ]
