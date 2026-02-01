"""Market data fetcher for stocks, crypto, and forex."""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
import yfinance as yf
import httpx
from ..config import settings


@dataclass
class OHLCV:
    """OHLCV data structure."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class MarketData:
    """Market data with metadata."""
    symbol: str
    name: str
    market_type: str  # 'stock', 'crypto', 'forex'
    current_price: float
    change_24h: float
    change_percent_24h: float
    high_24h: float
    low_24h: float
    volume_24h: float
    ohlcv: List[OHLCV]
    last_updated: datetime


class MarketDataFetcher:
    """Fetches market data from various sources."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    def _is_cache_valid(self, key: str, ttl: int = None) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache:
            return False
        if ttl is None:
            ttl = settings.CACHE_TTL_MARKET
        age = (datetime.now() - self._cache_timestamps.get(key, datetime.min)).total_seconds()
        return age < ttl

    def _set_cache(self, key: str, value: Any) -> None:
        """Set cached data."""
        self._cache[key] = value
        self._cache_timestamps[key] = datetime.now()

    def _get_cache(self, key: str) -> Any:
        """Get cached data."""
        return self._cache.get(key)

    def get_stock_data(
        self,
        symbols: List[str] = None,
        period: str = "1mo",
        interval: str = "1d"
    ) -> Dict[str, MarketData]:
        """
        Fetch stock data using yfinance.

        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL'])
            period: Data period ('1d', '5d', '1mo', '3mo', '1y')
            interval: Data interval ('1m', '5m', '1h', '1d')

        Returns:
            Dictionary mapping symbols to MarketData
        """
        symbols = symbols or settings.DEFAULT_STOCKS
        result = {}

        for symbol in symbols:
            cache_key = f"stock_{symbol}_{period}_{interval}"
            if self._is_cache_valid(cache_key):
                result[symbol] = self._get_cache(cache_key)
                continue

            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=interval)

                if hist.empty:
                    continue

                ohlcv_list = []
                for idx, row in hist.iterrows():
                    ohlcv_list.append(OHLCV(
                        timestamp=idx.to_pydatetime(),
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=float(row['Volume'])
                    ))

                current_price = float(hist['Close'].iloc[-1])
                prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
                change = current_price - prev_close
                change_percent = (change / prev_close * 100) if prev_close else 0

                info = ticker.info
                market_data = MarketData(
                    symbol=symbol,
                    name=info.get('shortName', symbol),
                    market_type='stock',
                    current_price=current_price,
                    change_24h=change,
                    change_percent_24h=change_percent,
                    high_24h=float(hist['High'].iloc[-1]),
                    low_24h=float(hist['Low'].iloc[-1]),
                    volume_24h=float(hist['Volume'].iloc[-1]),
                    ohlcv=ohlcv_list,
                    last_updated=datetime.now()
                )

                self._set_cache(cache_key, market_data)
                result[symbol] = market_data

            except Exception as e:
                print(f"Error fetching stock data for {symbol}: {e}")
                continue

        return result

    async def get_crypto_data(
        self,
        coins: List[str] = None,
        days: int = 30
    ) -> Dict[str, MarketData]:
        """
        Fetch cryptocurrency data from CoinGecko.

        Args:
            coins: List of coin IDs (e.g., ['bitcoin', 'ethereum'])
            days: Number of days of historical data

        Returns:
            Dictionary mapping coin IDs to MarketData
        """
        coins = coins or settings.DEFAULT_CRYPTO
        result = {}

        async with httpx.AsyncClient() as client:
            for coin in coins:
                cache_key = f"crypto_{coin}_{days}"
                if self._is_cache_valid(cache_key):
                    result[coin] = self._get_cache(cache_key)
                    continue

                try:
                    # Get current price data
                    price_url = f"{settings.COINGECKO_API_URL}/coins/{coin}"
                    price_resp = await client.get(price_url, timeout=10)
                    price_data = price_resp.json()

                    # Get historical data
                    hist_url = f"{settings.COINGECKO_API_URL}/coins/{coin}/market_chart"
                    hist_resp = await client.get(
                        hist_url,
                        params={"vs_currency": "usd", "days": days},
                        timeout=10
                    )
                    hist_data = hist_resp.json()

                    ohlcv_list = []
                    prices = hist_data.get('prices', [])
                    for i, (ts, price) in enumerate(prices):
                        ohlcv_list.append(OHLCV(
                            timestamp=datetime.fromtimestamp(ts / 1000),
                            open=price,
                            high=price,
                            low=price,
                            close=price,
                            volume=hist_data.get('total_volumes', [[0, 0]])[i][1] if i < len(hist_data.get('total_volumes', [])) else 0
                        ))

                    market_info = price_data.get('market_data', {})
                    market_data = MarketData(
                        symbol=coin.upper(),
                        name=price_data.get('name', coin),
                        market_type='crypto',
                        current_price=market_info.get('current_price', {}).get('usd', 0),
                        change_24h=market_info.get('price_change_24h', 0),
                        change_percent_24h=market_info.get('price_change_percentage_24h', 0),
                        high_24h=market_info.get('high_24h', {}).get('usd', 0),
                        low_24h=market_info.get('low_24h', {}).get('usd', 0),
                        volume_24h=market_info.get('total_volume', {}).get('usd', 0),
                        ohlcv=ohlcv_list,
                        last_updated=datetime.now()
                    )

                    self._set_cache(cache_key, market_data)
                    result[coin] = market_data

                except Exception as e:
                    print(f"Error fetching crypto data for {coin}: {e}")
                    continue

                # Rate limiting for free API
                await asyncio.sleep(0.5)

        return result

    async def get_forex_data(
        self,
        currencies: List[str] = None,
        base: str = "USD"
    ) -> Dict[str, MarketData]:
        """
        Fetch forex data from ExchangeRate API.

        Args:
            currencies: List of currency codes (e.g., ['EUR', 'GBP'])
            base: Base currency for rates

        Returns:
            Dictionary mapping currency pairs to MarketData
        """
        currencies = currencies or settings.DEFAULT_FOREX
        result = {}

        cache_key = f"forex_{base}"
        if self._is_cache_valid(cache_key):
            cached_data = self._get_cache(cache_key)
            return {k: v for k, v in cached_data.items() if k in currencies}

        try:
            async with httpx.AsyncClient() as client:
                url = f"{settings.EXCHANGERATE_API_URL}/{base}"
                resp = await client.get(url, timeout=10)
                data = resp.json()

                rates = data.get('rates', {})
                for currency in currencies:
                    if currency not in rates:
                        continue

                    rate = rates[currency]
                    market_data = MarketData(
                        symbol=f"{base}/{currency}",
                        name=f"{base} to {currency}",
                        market_type='forex',
                        current_price=rate,
                        change_24h=0,  # Free API doesn't provide historical
                        change_percent_24h=0,
                        high_24h=rate,
                        low_24h=rate,
                        volume_24h=0,
                        ohlcv=[],
                        last_updated=datetime.now()
                    )
                    result[currency] = market_data

                self._set_cache(cache_key, result)

        except Exception as e:
            print(f"Error fetching forex data: {e}")

        return result

    async def get_all_market_data(self) -> Dict[str, Dict[str, MarketData]]:
        """
        Fetch all market data (stocks, crypto, forex).

        Returns:
            Dictionary with 'stocks', 'crypto', 'forex' keys
        """
        crypto_task = self.get_crypto_data()
        forex_task = self.get_forex_data()

        # Run stock fetch in thread pool (yfinance is sync)
        loop = asyncio.get_event_loop()
        stocks = await loop.run_in_executor(None, self.get_stock_data)

        crypto, forex = await asyncio.gather(crypto_task, forex_task)

        return {
            'stocks': stocks,
            'crypto': crypto,
            'forex': forex
        }

    def get_price_series(self, market_data: MarketData) -> pd.Series:
        """Convert MarketData OHLCV to a pandas Series of closing prices."""
        if not market_data.ohlcv:
            return pd.Series(dtype=float)

        timestamps = [o.timestamp for o in market_data.ohlcv]
        closes = [o.close for o in market_data.ohlcv]
        return pd.Series(closes, index=pd.DatetimeIndex(timestamps), name=market_data.symbol)
