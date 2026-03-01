"""XAUT/USDT price stream using CoinGecko free API."""

import asyncio
import sys
from pathlib import Path
import time
import requests
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.event_bus import Event, EventType, get_event_bus
from quant.core.config import get_config
from quant.core.logger import get_logger

logger = get_logger(__name__)

COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"


@dataclass
class XAUTData:
    price: float
    change_pct_24h: float
    premium_discount_pct: float  # vs XAU spot
    xau_spot: float              # latest known XAU spot for comparison
    timestamp: float = field(default_factory=time.time)


class XAUTStream:
    """Streams XAUT/USDT price from CoinGecko and computes premium/discount vs XAU spot."""

    def __init__(self, xau_price_ref: list):
        """
        Args:
            xau_price_ref: A mutable list [price] holding the latest XAU spot price,
                           updated by XAUStream. Shared by reference.
        """
        self.config = get_config()
        self.bus = get_event_bus()
        self._xau_ref = xau_price_ref
        self._last_price: Optional[float] = None

    def _fetch_xaut(self) -> Optional[dict]:
        """Fetch XAUT price from CoinGecko."""
        try:
            resp = requests.get(
                COINGECKO_URL,
                params={
                    "ids": "tether-gold",
                    "vs_currencies": "usd",
                    "include_24hr_change": "true",
                },
                headers={"User-Agent": "Mozilla/5.0 (compatible; QuantBot/1.0)"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            tether_gold = data.get("tether-gold", {})
            price = tether_gold.get("usd")
            change_24h = tether_gold.get("usd_24h_change", 0.0)
            if price:
                return {"price": float(price), "change_24h": float(change_24h or 0)}
        except Exception as e:
            logger.warning("xaut_fetch_failed", error=str(e))
        return None

    async def run(self) -> None:
        """Main stream loop."""
        logger.info("xaut_stream_started")
        while True:
            try:
                result = await asyncio.get_event_loop().run_in_executor(None, self._fetch_xaut)
                if result:
                    price = result["price"]
                    change_24h = result["change_24h"]
                    xau_spot = self._xau_ref[0] if self._xau_ref else price
                    premium_pct = 0.0
                    if xau_spot and xau_spot > 0:
                        premium_pct = ((price - xau_spot) / xau_spot) * 100

                    data = XAUTData(
                        price=price,
                        change_pct_24h=change_24h,
                        premium_discount_pct=premium_pct,
                        xau_spot=xau_spot,
                    )
                    self._last_price = price

                    event = Event(type=EventType.XAUT_PRICE, data=data, source="xaut_stream")
                    await self.bus.publish(event)
                    logger.debug("xaut_price_published", price=price, premium_pct=round(premium_pct, 4))

            except asyncio.CancelledError:
                logger.info("xaut_stream_stopped")
                break
            except Exception as e:
                logger.error("xaut_stream_error", error=str(e))

            await asyncio.sleep(self.config.xaut_poll_interval)
