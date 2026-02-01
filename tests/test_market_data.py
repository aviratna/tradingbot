"""Tests for market data fetcher module."""
import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from app.data.market_data import (
    MarketDataFetcher,
    MarketData,
    OHLCV
)


@pytest.fixture
def fetcher():
    """Create a market data fetcher instance."""
    return MarketDataFetcher()


@pytest.fixture
def mock_stock_data():
    """Create mock stock data."""
    return MarketData(
        symbol="AAPL",
        name="Apple Inc.",
        market_type="stock",
        current_price=175.50,
        change_24h=2.30,
        change_percent_24h=1.33,
        high_24h=176.20,
        low_24h=173.80,
        volume_24h=45000000,
        ohlcv=[
            OHLCV(
                timestamp=datetime(2024, 1, 1),
                open=173.00,
                high=176.20,
                low=172.50,
                close=175.50,
                volume=45000000
            )
        ],
        last_updated=datetime.now()
    )


class TestOHLCV:
    """Tests for OHLCV dataclass."""

    def test_ohlcv_creation(self):
        """Test creating OHLCV data."""
        ohlcv = OHLCV(
            timestamp=datetime(2024, 1, 1, 9, 30),
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            volume=1000000
        )

        assert ohlcv.open == 100.0
        assert ohlcv.high == 105.0
        assert ohlcv.low == 99.0
        assert ohlcv.close == 104.0
        assert ohlcv.volume == 1000000


class TestMarketData:
    """Tests for MarketData dataclass."""

    def test_market_data_creation(self, mock_stock_data):
        """Test creating MarketData."""
        assert mock_stock_data.symbol == "AAPL"
        assert mock_stock_data.market_type == "stock"
        assert mock_stock_data.current_price == 175.50

    def test_market_data_has_ohlcv(self, mock_stock_data):
        """Test that MarketData contains OHLCV list."""
        assert isinstance(mock_stock_data.ohlcv, list)
        assert len(mock_stock_data.ohlcv) == 1
        assert isinstance(mock_stock_data.ohlcv[0], OHLCV)


class TestCaching:
    """Tests for caching functionality."""

    def test_cache_not_valid_initially(self, fetcher):
        """Test that cache is not valid initially."""
        assert not fetcher._is_cache_valid("nonexistent_key")

    def test_cache_set_and_get(self, fetcher):
        """Test setting and getting cached data."""
        fetcher._set_cache("test_key", {"data": "value"})

        assert fetcher._is_cache_valid("test_key")
        assert fetcher._get_cache("test_key") == {"data": "value"}

    def test_cache_expiry(self, fetcher):
        """Test that cache respects TTL."""
        fetcher._set_cache("test_key", {"data": "value"})

        # Cache should be valid with default TTL
        assert fetcher._is_cache_valid("test_key", ttl=60)

        # But not with TTL of 0
        assert not fetcher._is_cache_valid("test_key", ttl=0)


class TestGetPriceSeries:
    """Tests for price series conversion."""

    def test_get_price_series_basic(self, fetcher, mock_stock_data):
        """Test converting MarketData to price series."""
        series = fetcher.get_price_series(mock_stock_data)

        assert isinstance(series, pd.Series)
        assert len(series) == 1
        assert series.iloc[0] == 175.50
        assert series.name == "AAPL"

    def test_get_price_series_empty(self, fetcher):
        """Test price series with no OHLCV data."""
        market_data = MarketData(
            symbol="EMPTY",
            name="Empty Data",
            market_type="stock",
            current_price=0,
            change_24h=0,
            change_percent_24h=0,
            high_24h=0,
            low_24h=0,
            volume_24h=0,
            ohlcv=[],
            last_updated=datetime.now()
        )

        series = fetcher.get_price_series(market_data)

        assert isinstance(series, pd.Series)
        assert len(series) == 0


class TestStockDataFetcher:
    """Tests for stock data fetching."""

    @patch('yfinance.Ticker')
    def test_get_stock_data_success(self, mock_ticker, fetcher):
        """Test successful stock data fetch."""
        # Setup mock
        mock_history = pd.DataFrame({
            'Open': [173.0, 174.0],
            'High': [176.0, 177.0],
            'Low': [172.0, 173.0],
            'Close': [175.0, 176.0],
            'Volume': [45000000, 46000000]
        }, index=pd.date_range('2024-01-01', periods=2, freq='D'))

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_history
        mock_ticker_instance.info = {'shortName': 'Apple Inc.'}
        mock_ticker.return_value = mock_ticker_instance

        result = fetcher.get_stock_data(['AAPL'], period='1mo')

        assert 'AAPL' in result
        assert result['AAPL'].symbol == 'AAPL'
        assert result['AAPL'].market_type == 'stock'

    @patch('yfinance.Ticker')
    def test_get_stock_data_empty_history(self, mock_ticker, fetcher):
        """Test stock data with empty history."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance

        result = fetcher.get_stock_data(['INVALID'])

        assert 'INVALID' not in result

    @patch('yfinance.Ticker')
    def test_get_stock_data_exception(self, mock_ticker, fetcher):
        """Test stock data handling exception."""
        mock_ticker.side_effect = Exception("API Error")

        result = fetcher.get_stock_data(['ERROR'])

        assert 'ERROR' not in result


class TestCryptoDataFetcher:
    """Tests for crypto data fetching."""

    @pytest.mark.asyncio
    async def test_get_crypto_data_success(self, fetcher):
        """Test successful crypto data fetch with mocked response."""
        mock_price_response = {
            'name': 'Bitcoin',
            'market_data': {
                'current_price': {'usd': 45000},
                'price_change_24h': 500,
                'price_change_percentage_24h': 1.12,
                'high_24h': {'usd': 46000},
                'low_24h': {'usd': 44000},
                'total_volume': {'usd': 25000000000}
            }
        }

        mock_history_response = {
            'prices': [[1704067200000, 45000], [1704153600000, 45500]],
            'total_volumes': [[1704067200000, 25000000000], [1704153600000, 26000000000]]
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance

            price_resp = AsyncMock()
            price_resp.json.return_value = mock_price_response

            hist_resp = AsyncMock()
            hist_resp.json.return_value = mock_history_response

            mock_instance.get.side_effect = [price_resp, hist_resp]

            result = await fetcher.get_crypto_data(['bitcoin'], days=30)

            assert 'bitcoin' in result
            assert result['bitcoin'].market_type == 'crypto'

    @pytest.mark.asyncio
    async def test_get_crypto_data_exception(self, fetcher):
        """Test crypto data handling exception."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.get.side_effect = Exception("API Error")

            result = await fetcher.get_crypto_data(['bitcoin'])

            assert 'bitcoin' not in result


class TestForexDataFetcher:
    """Tests for forex data fetching."""

    @pytest.mark.asyncio
    async def test_get_forex_data_success(self, fetcher):
        """Test successful forex data fetch."""
        mock_response = {
            'rates': {
                'EUR': 0.92,
                'GBP': 0.79,
                'JPY': 148.50
            }
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance

            resp = AsyncMock()
            resp.json.return_value = mock_response
            mock_instance.get.return_value = resp

            result = await fetcher.get_forex_data(['EUR', 'GBP'], base='USD')

            assert 'EUR' in result
            assert 'GBP' in result
            assert result['EUR'].market_type == 'forex'
            assert result['EUR'].current_price == 0.92

    @pytest.mark.asyncio
    async def test_get_forex_data_missing_currency(self, fetcher):
        """Test forex data with missing currency."""
        mock_response = {
            'rates': {
                'EUR': 0.92
            }
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance

            resp = AsyncMock()
            resp.json.return_value = mock_response
            mock_instance.get.return_value = resp

            result = await fetcher.get_forex_data(['EUR', 'INVALID'])

            assert 'EUR' in result
            assert 'INVALID' not in result


class TestGetAllMarketData:
    """Tests for fetching all market data."""

    @pytest.mark.asyncio
    async def test_get_all_market_data(self, fetcher):
        """Test fetching all market data types."""
        with patch.object(fetcher, 'get_stock_data') as mock_stocks, \
             patch.object(fetcher, 'get_crypto_data', new_callable=AsyncMock) as mock_crypto, \
             patch.object(fetcher, 'get_forex_data', new_callable=AsyncMock) as mock_forex:

            mock_stocks.return_value = {'AAPL': Mock()}
            mock_crypto.return_value = {'bitcoin': Mock()}
            mock_forex.return_value = {'EUR': Mock()}

            result = await fetcher.get_all_market_data()

            assert 'stocks' in result
            assert 'crypto' in result
            assert 'forex' in result


class TestDefaultSymbols:
    """Tests for default symbol handling."""

    def test_default_stocks_used(self, fetcher):
        """Test that default stocks are used when none provided."""
        from app.config import settings

        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = pd.DataFrame()
            mock_ticker.return_value = mock_ticker_instance

            fetcher.get_stock_data()  # No symbols provided

            # Should have called for each default stock
            assert mock_ticker.call_count == len(settings.DEFAULT_STOCKS)
