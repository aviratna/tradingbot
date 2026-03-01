"""XAU/USD spot price stream using goldprice.org and yfinance fallback."""

import asyncio
import sys
from pathlib import Path
import time
import requests
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.event_bus import Event, EventType, get_event_bus
from quant.core.config import get_config
from quant.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class XAUBar:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class XAUData:
    price: float
    change_pct: float
    prev_price: float
    bars: list  # List of XAUBar
    timestamp: float = field(default_factory=time.time)


class XAUStream:
    """Streams XAU/USD spot price, building rolling OHLCV bars."""

    GOLDPRICE_URL = "https://data-asg.goldprice.org/dbXRates/USD"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; QuantBot/1.0)",
        "Accept": "application/json",
        "Referer": "https://goldprice.org",
    }

    def __init__(self):
        self.config = get_config()
        self.bus = get_event_bus()
        self._bars: Deque[XAUBar] = deque(maxlen=500)
        self._last_price: Optional[float] = None
        self._prev_close: Optional[float] = None
        self._bar_open: Optional[float] = None
        self._bar_high: float = 0.0
        self._bar_low: float = float("inf")
        self._bar_start: float = 0.0
        self._bar_duration: float = 300.0  # 5-minute bars

    def _fetch_price(self) -> Optional[float]:
        """Fetch current XAU/USD price from goldprice.org."""
        try:
            resp = requests.get(self.GOLDPRICE_URL, headers=self.HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            # goldprice.org returns price per troy ounce in USD
            items = data.get("items", [])
            if items:
                price = float(items[0].get("xauPrice", 0))
                if price > 0:
                    return price
        except Exception as e:
            logger.warning("goldprice_fetch_failed", error=str(e))

        # Fallback: yfinance GLD ETF × scale factor
        return self._fetch_via_yfinance()

    def _fetch_via_yfinance(self) -> Optional[float]:
        """Fallback: get GLD ETF price and scale to spot gold."""
        try:
            import yfinance as yf
            gld = yf.Ticker("GLD")
            hist = gld.history(period="1d", interval="1m")
            if not hist.empty:
                etf_price = float(hist["Close"].iloc[-1])
                # GLD: 1 share ≈ 0.09357 troy oz of gold
                spot = etf_price / 0.09357
                logger.debug("xau_yfinance_fallback", etf_price=etf_price, spot=spot)
                return spot
        except Exception as e:
            logger.warning("xau_yfinance_fallback_failed", error=str(e))
        return None

    def _update_bar(self, price: float, ts: float) -> Optional[XAUBar]:
        """Update current 5-min bar, return completed bar if interval elapsed."""
        completed = None
        if self._bar_open is None or (ts - self._bar_start) >= self._bar_duration:
            if self._bar_open is not None:
                # Complete the current bar
                completed = XAUBar(
                    timestamp=self._bar_start,
                    open=self._bar_open,
                    high=self._bar_high,
                    low=self._bar_low,
                    close=self._last_price or price,
                )
                self._bars.append(completed)
            # Start new bar
            self._bar_open = price
            self._bar_high = price
            self._bar_low = price
            self._bar_start = ts
        else:
            self._bar_high = max(self._bar_high, price)
            self._bar_low = min(self._bar_low, price)
        return completed

    async def run(self) -> None:
        """Main stream loop."""
        logger.info("xau_stream_started")
        # Pre-populate bars with historical data
        await self._preload_bars()

        while True:
            try:
                price = await asyncio.get_event_loop().run_in_executor(None, self._fetch_price)
                if price and price > 0:
                    ts = time.time()
                    self._update_bar(price, ts)
                    change_pct = 0.0
                    if self._prev_close and self._prev_close > 0:
                        change_pct = ((price - self._prev_close) / self._prev_close) * 100
                    elif self._last_price and self._last_price > 0:
                        change_pct = ((price - self._last_price) / self._last_price) * 100

                    data = XAUData(
                        price=price,
                        change_pct=change_pct,
                        prev_price=self._last_price or price,
                        bars=list(self._bars),
                        timestamp=ts,
                    )
                    self._last_price = price

                    event = Event(type=EventType.XAU_PRICE, data=data, source="xau_stream")
                    await self.bus.publish(event)
                    logger.debug("xau_price_published", price=price, change_pct=round(change_pct, 4))
                else:
                    logger.warning("xau_price_invalid", price=price)

            except asyncio.CancelledError:
                logger.info("xau_stream_stopped")
                break
            except Exception as e:
                logger.error("xau_stream_error", error=str(e))

            await asyncio.sleep(self.config.xau_poll_interval)

    async def _preload_bars(self) -> None:
        """Pre-load historical OHLCV bars from yfinance."""
        try:
            import yfinance as yf
            def _fetch():
                gld = yf.Ticker("GLD")
                hist = gld.history(period="5d", interval="5m")
                return hist

            hist = await asyncio.get_event_loop().run_in_executor(None, _fetch)
            if not hist.empty:
                scale = 1 / 0.09357  # GLD to spot gold
                for _, row in hist.iterrows():
                    bar = XAUBar(
                        timestamp=row.name.timestamp(),
                        open=float(row["Open"]) * scale,
                        high=float(row["High"]) * scale,
                        low=float(row["Low"]) * scale,
                        close=float(row["Close"]) * scale,
                        volume=float(row.get("Volume", 0)),
                    )
                    self._bars.append(bar)
                if self._bars:
                    self._prev_close = self._bars[-1].close
                logger.info("xau_bars_preloaded", count=len(self._bars))
        except Exception as e:
            logger.warning("xau_preload_failed", error=str(e))
