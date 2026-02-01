"""
Precious Metals Data Fetcher for XAU/USD and XAG/USD trading.
Provides real-time and historical data for gold and silver with session detection.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional
import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class TradingSession(Enum):
    """Trading session enumeration with manipulation risk levels."""
    ASIAN = "asian"
    LONDON = "london"
    NY_OVERLAP = "ny_overlap"
    NY = "new_york"
    NY_CLOSE = "ny_close"
    WEEKEND = "weekend"


@dataclass
class SessionInfo:
    """Information about current trading session."""
    session: TradingSession
    name: str
    start_hour: int
    end_hour: int
    manipulation_risk: str  # LOW, MODERATE, HIGH, HIGHEST
    description: str
    is_active: bool = False
    time_remaining: Optional[timedelta] = None


@dataclass
class OHLCV:
    """OHLCV candlestick data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class SpreadInfo:
    """Spread monitoring data."""
    bid: float
    ask: float
    spread_pips: float
    spread_percent: float
    is_wide: bool  # True if spread is unusually wide


@dataclass
class VolumeAnalysis:
    """Volume analysis data."""
    current_volume: float
    avg_volume: float
    volume_ratio: float  # Current / Average
    is_high_volume: bool
    is_low_volume: bool
    volume_trend: str  # INCREASING, DECREASING, STABLE


@dataclass
class MetalPrice:
    """Current precious metal price data."""
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    high_24h: float
    low_24h: float
    volume: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass
class MetalData:
    """Complete precious metal market data."""
    symbol: str
    name: str
    current_price: MetalPrice
    ohlcv_1m: list[OHLCV] = field(default_factory=list)
    ohlcv_5m: list[OHLCV] = field(default_factory=list)
    ohlcv_15m: list[OHLCV] = field(default_factory=list)
    ohlcv_1h: list[OHLCV] = field(default_factory=list)
    spread_info: Optional[SpreadInfo] = None
    volume_analysis: Optional[VolumeAnalysis] = None
    session_info: Optional[SessionInfo] = None


@dataclass
class CorrelatedAsset:
    """Data for correlated asset."""
    symbol: str
    name: str
    price: float
    change_percent: float
    correlation_type: str  # POSITIVE, NEGATIVE, INVERSE


class PreciousMetalsDataFetcher:
    """
    Fetches and processes precious metals data for XAU/USD and XAG/USD.
    Uses yfinance for gold (GC=F) and silver (SI=F) futures.
    """

    # Symbol mappings
    METALS = {
        "XAU/USD": {"yf_symbol": "GC=F", "name": "Gold", "pip_value": 0.01},
        "XAG/USD": {"yf_symbol": "SI=F", "name": "Silver", "pip_value": 0.001},
    }

    # Correlated assets for monitoring
    CORRELATED_ASSETS = {
        "DXY": {"yf_symbol": "DX-Y.NYB", "name": "US Dollar Index", "relationship": "INVERSE"},
        "US10Y": {"yf_symbol": "^TNX", "name": "10-Year Treasury Yield", "relationship": "INVERSE"},
        "SPY": {"yf_symbol": "SPY", "name": "S&P 500 ETF", "relationship": "NEGATIVE"},
        "VIX": {"yf_symbol": "^VIX", "name": "Volatility Index", "relationship": "POSITIVE"},
        "USO": {"yf_symbol": "USO", "name": "Crude Oil ETF", "relationship": "POSITIVE"},
        "EUR/USD": {"yf_symbol": "EURUSD=X", "name": "Euro/Dollar", "relationship": "POSITIVE"},
        "GBP/USD": {"yf_symbol": "GBPUSD=X", "name": "Pound/Dollar", "relationship": "POSITIVE"},
    }

    # Session definitions (UTC times)
    SESSIONS = {
        TradingSession.ASIAN: SessionInfo(
            session=TradingSession.ASIAN,
            name="Asian Session",
            start_hour=0,
            end_hour=8,
            manipulation_risk="LOW",
            description="Low volume, range-bound trading"
        ),
        TradingSession.LONDON: SessionInfo(
            session=TradingSession.LONDON,
            name="London Session",
            start_hour=8,
            end_hour=13,
            manipulation_risk="HIGH",
            description="High liquidity grabs, stop hunts common"
        ),
        TradingSession.NY_OVERLAP: SessionInfo(
            session=TradingSession.NY_OVERLAP,
            name="London/NY Overlap",
            start_hour=13,
            end_hour=14,
            manipulation_risk="HIGHEST",
            description="Peak manipulation - stop hunts at session open"
        ),
        TradingSession.NY: SessionInfo(
            session=TradingSession.NY,
            name="New York Session",
            start_hour=14,
            end_hour=20,
            manipulation_risk="MODERATE",
            description="High volume, trend continuation"
        ),
        TradingSession.NY_CLOSE: SessionInfo(
            session=TradingSession.NY_CLOSE,
            name="NY Close",
            start_hour=20,
            end_hour=24,
            manipulation_risk="MODERATE",
            description="Position squaring, potential reversals"
        ),
    }

    def __init__(self, cache_ttl: int = 60):
        """Initialize with optional cache TTL in seconds."""
        self._cache: dict = {}
        self._cache_ttl = cache_ttl

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache:
            return False
        cached_time = self._cache[key].get("timestamp")
        if not cached_time:
            return False
        return (datetime.now(timezone.utc) - cached_time).seconds < self._cache_ttl

    def get_current_session(self) -> SessionInfo:
        """Get the current trading session based on UTC time."""
        now = datetime.now(timezone.utc)
        current_hour = now.hour

        # Check if weekend
        if now.weekday() >= 5:  # Saturday or Sunday
            return SessionInfo(
                session=TradingSession.WEEKEND,
                name="Weekend",
                start_hour=0,
                end_hour=24,
                manipulation_risk="N/A",
                description="Market closed",
                is_active=False
            )

        for session_enum, session_info in self.SESSIONS.items():
            if session_info.start_hour <= current_hour < session_info.end_hour:
                # Calculate time remaining
                end_time = now.replace(hour=session_info.end_hour, minute=0, second=0)
                if session_info.end_hour == 24:
                    end_time = now.replace(hour=23, minute=59, second=59)
                time_remaining = end_time - now

                return SessionInfo(
                    session=session_info.session,
                    name=session_info.name,
                    start_hour=session_info.start_hour,
                    end_hour=session_info.end_hour,
                    manipulation_risk=session_info.manipulation_risk,
                    description=session_info.description,
                    is_active=True,
                    time_remaining=time_remaining
                )

        # Default to Asian session (covers edge case)
        return self.SESSIONS[TradingSession.ASIAN]

    def _fetch_ohlcv(
        self,
        yf_symbol: str,
        period: str = "5d",
        interval: str = "1m"
    ) -> list[OHLCV]:
        """Fetch OHLCV data from yfinance."""
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {yf_symbol}")
                return []

            ohlcv_list = []
            for idx, row in df.iterrows():
                ohlcv_list.append(OHLCV(
                    timestamp=idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx,
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row['Volume']) if not pd.isna(row['Volume']) else 0.0
                ))

            return ohlcv_list

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {yf_symbol}: {e}")
            return []

    def _analyze_volume(self, ohlcv: list[OHLCV], lookback: int = 20) -> VolumeAnalysis:
        """Analyze volume patterns."""
        if not ohlcv or len(ohlcv) < 2:
            return VolumeAnalysis(
                current_volume=0,
                avg_volume=0,
                volume_ratio=0,
                is_high_volume=False,
                is_low_volume=False,
                volume_trend="STABLE"
            )

        volumes = [candle.volume for candle in ohlcv[-lookback:] if candle.volume > 0]

        if not volumes:
            return VolumeAnalysis(
                current_volume=0,
                avg_volume=0,
                volume_ratio=0,
                is_high_volume=False,
                is_low_volume=False,
                volume_trend="STABLE"
            )

        current_volume = volumes[-1] if volumes else 0
        avg_volume = np.mean(volumes)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        # Determine trend
        if len(volumes) >= 5:
            recent_avg = np.mean(volumes[-5:])
            older_avg = np.mean(volumes[:-5]) if len(volumes) > 5 else avg_volume
            if recent_avg > older_avg * 1.2:
                trend = "INCREASING"
            elif recent_avg < older_avg * 0.8:
                trend = "DECREASING"
            else:
                trend = "STABLE"
        else:
            trend = "STABLE"

        return VolumeAnalysis(
            current_volume=current_volume,
            avg_volume=avg_volume,
            volume_ratio=volume_ratio,
            is_high_volume=volume_ratio > 1.5,
            is_low_volume=volume_ratio < 0.5,
            volume_trend=trend
        )

    def _calculate_spread(self, ohlcv: list[OHLCV], pip_value: float) -> SpreadInfo:
        """Calculate spread information (estimated from OHLCV)."""
        if not ohlcv:
            return SpreadInfo(
                bid=0, ask=0, spread_pips=0, spread_percent=0, is_wide=False
            )

        # Estimate spread from high-low range of most recent candles
        recent_candles = ohlcv[-5:] if len(ohlcv) >= 5 else ohlcv
        avg_range = np.mean([c.high - c.low for c in recent_candles])

        # Estimate bid/ask from close
        last_close = ohlcv[-1].close
        estimated_spread = avg_range * 0.1  # Rough estimate

        bid = last_close - estimated_spread / 2
        ask = last_close + estimated_spread / 2
        spread_pips = estimated_spread / pip_value
        spread_percent = (estimated_spread / last_close) * 100 if last_close > 0 else 0

        # Wide spread threshold (varies by asset)
        is_wide = spread_pips > 5  # More than 5 pips is considered wide

        return SpreadInfo(
            bid=bid,
            ask=ask,
            spread_pips=spread_pips,
            spread_percent=spread_percent,
            is_wide=is_wide
        )

    def get_metal_data(self, symbol: str = "XAU/USD") -> MetalData:
        """
        Get comprehensive data for a precious metal.

        Args:
            symbol: Metal symbol (XAU/USD or XAG/USD)

        Returns:
            MetalData with all timeframes and analysis
        """
        if symbol not in self.METALS:
            raise ValueError(f"Unknown metal symbol: {symbol}. Use XAU/USD or XAG/USD")

        metal_info = self.METALS[symbol]
        yf_symbol = metal_info["yf_symbol"]

        # Check cache
        cache_key = f"metal_{symbol}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]["data"]

        # Fetch different timeframes
        ohlcv_1m = self._fetch_ohlcv(yf_symbol, period="1d", interval="1m")
        ohlcv_5m = self._fetch_ohlcv(yf_symbol, period="5d", interval="5m")
        ohlcv_15m = self._fetch_ohlcv(yf_symbol, period="5d", interval="15m")
        ohlcv_1h = self._fetch_ohlcv(yf_symbol, period="1mo", interval="1h")

        # Get current price from most recent data
        price_data = ohlcv_1m if ohlcv_1m else ohlcv_5m

        if price_data:
            current = price_data[-1]
            # Calculate 24h change
            day_ago_idx = max(0, len(price_data) - 390)  # ~6.5 hours of 1m candles
            day_ago_price = price_data[day_ago_idx].close if len(price_data) > day_ago_idx else current.open
            change = current.close - day_ago_price
            change_percent = (change / day_ago_price) * 100 if day_ago_price > 0 else 0

            # 24h high/low
            high_24h = max(c.high for c in price_data)
            low_24h = min(c.low for c in price_data)

            current_price = MetalPrice(
                symbol=symbol,
                name=metal_info["name"],
                price=current.close,
                change=change,
                change_percent=change_percent,
                high_24h=high_24h,
                low_24h=low_24h,
                volume=current.volume,
                timestamp=current.timestamp
            )
        else:
            # Fallback to basic fetch
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            current_price = MetalPrice(
                symbol=symbol,
                name=metal_info["name"],
                price=info.get("regularMarketPrice", 0),
                change=info.get("regularMarketChange", 0),
                change_percent=info.get("regularMarketChangePercent", 0),
                high_24h=info.get("dayHigh", 0),
                low_24h=info.get("dayLow", 0),
                volume=info.get("volume", 0),
                timestamp=datetime.now(timezone.utc)
            )

        # Analyze volume
        volume_analysis = self._analyze_volume(ohlcv_1m if ohlcv_1m else ohlcv_5m)

        # Calculate spread
        spread_info = self._calculate_spread(
            ohlcv_1m if ohlcv_1m else ohlcv_5m,
            metal_info["pip_value"]
        )

        # Get session info
        session_info = self.get_current_session()

        metal_data = MetalData(
            symbol=symbol,
            name=metal_info["name"],
            current_price=current_price,
            ohlcv_1m=ohlcv_1m,
            ohlcv_5m=ohlcv_5m,
            ohlcv_15m=ohlcv_15m,
            ohlcv_1h=ohlcv_1h,
            spread_info=spread_info,
            volume_analysis=volume_analysis,
            session_info=session_info
        )

        # Cache the data
        self._cache[cache_key] = {
            "data": metal_data,
            "timestamp": datetime.now(timezone.utc)
        }

        return metal_data

    def get_all_metals(self) -> dict[str, MetalData]:
        """Get data for all precious metals (XAU/USD and XAG/USD)."""
        return {
            symbol: self.get_metal_data(symbol)
            for symbol in self.METALS.keys()
        }

    def get_correlated_assets(self) -> list[CorrelatedAsset]:
        """Fetch data for all correlated assets."""
        assets = []

        for asset_key, asset_info in self.CORRELATED_ASSETS.items():
            try:
                ticker = yf.Ticker(asset_info["yf_symbol"])
                hist = ticker.history(period="2d", interval="1h")

                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[0]
                    change_percent = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0

                    assets.append(CorrelatedAsset(
                        symbol=asset_key,
                        name=asset_info["name"],
                        price=current_price,
                        change_percent=change_percent,
                        correlation_type=asset_info["relationship"]
                    ))
            except Exception as e:
                logger.warning(f"Failed to fetch {asset_key}: {e}")
                continue

        return assets

    async def get_metal_data_async(self, symbol: str = "XAU/USD") -> MetalData:
        """Async wrapper for get_metal_data."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_metal_data, symbol)

    async def get_all_metals_async(self) -> dict[str, MetalData]:
        """Async wrapper for get_all_metals."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_all_metals)

    async def get_correlated_assets_async(self) -> list[CorrelatedAsset]:
        """Async wrapper for get_correlated_assets."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_correlated_assets)

    def get_price_series(
        self,
        symbol: str = "XAU/USD",
        timeframe: str = "5m"
    ) -> pd.Series:
        """
        Get price series for technical analysis.

        Args:
            symbol: Metal symbol
            timeframe: 1m, 5m, 15m, or 1h

        Returns:
            Pandas Series with close prices indexed by timestamp
        """
        metal_data = self.get_metal_data(symbol)

        timeframe_map = {
            "1m": metal_data.ohlcv_1m,
            "5m": metal_data.ohlcv_5m,
            "15m": metal_data.ohlcv_15m,
            "1h": metal_data.ohlcv_1h,
        }

        ohlcv = timeframe_map.get(timeframe, metal_data.ohlcv_5m)

        if not ohlcv:
            return pd.Series(dtype=float)

        return pd.Series(
            [c.close for c in ohlcv],
            index=[c.timestamp for c in ohlcv],
            name=f"{symbol}_{timeframe}"
        )

    def get_ohlcv_dataframe(
        self,
        symbol: str = "XAU/USD",
        timeframe: str = "5m"
    ) -> pd.DataFrame:
        """
        Get OHLCV data as a pandas DataFrame.

        Args:
            symbol: Metal symbol
            timeframe: 1m, 5m, 15m, or 1h

        Returns:
            DataFrame with OHLCV columns
        """
        metal_data = self.get_metal_data(symbol)

        timeframe_map = {
            "1m": metal_data.ohlcv_1m,
            "5m": metal_data.ohlcv_5m,
            "15m": metal_data.ohlcv_15m,
            "1h": metal_data.ohlcv_1h,
        }

        ohlcv = timeframe_map.get(timeframe, metal_data.ohlcv_5m)

        if not ohlcv:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume
            }
            for c in ohlcv
        ]).set_index("timestamp")


# Singleton instance for convenience
precious_metals_fetcher = PreciousMetalsDataFetcher()
