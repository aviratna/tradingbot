"""Macro asset stream: DXY, US10Y, SPY, VIX, USO via yfinance."""

import asyncio
import sys
from pathlib import Path
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
import yfinance as yf
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.event_bus import Event, EventType, get_event_bus
from quant.core.config import get_config
from quant.core.logger import get_logger

logger = get_logger(__name__)

MACRO_TICKERS = {
    "DXY": "DX-Y.NYB",
    "US10Y": "^TNX",
    "SPY": "SPY",
    "VIX": "^VIX",
    "USO": "USO",
}


@dataclass
class MacroAsset:
    symbol: str
    price: float
    change_pct_1d: float
    change_pct_5d: float
    price_history: list  # list of closing prices (recent 30 days)


@dataclass
class MacroData:
    dxy: Optional[MacroAsset] = None
    us10y: Optional[MacroAsset] = None
    spy: Optional[MacroAsset] = None
    vix: Optional[MacroAsset] = None
    uso: Optional[MacroAsset] = None
    timestamp: float = field(default_factory=time.time)


class MacroStream:
    """Polls macro assets every 60 seconds via yfinance."""

    def __init__(self):
        self.config = get_config()
        self.bus = get_event_bus()

    def _fetch_macro(self) -> MacroData:
        """Fetch all macro tickers synchronously."""
        data = MacroData()
        try:
            tickers = yf.download(
                list(MACRO_TICKERS.values()),
                period="32d",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            close = tickers["Close"] if "Close" in tickers.columns else tickers

            for key, ticker in MACRO_TICKERS.items():
                try:
                    if ticker not in close.columns:
                        continue
                    series = close[ticker].dropna()
                    if len(series) < 2:
                        continue
                    current = float(series.iloc[-1])
                    prev_1d = float(series.iloc[-2]) if len(series) >= 2 else current
                    prev_5d = float(series.iloc[-6]) if len(series) >= 6 else float(series.iloc[0])
                    change_1d = ((current - prev_1d) / prev_1d) * 100 if prev_1d != 0 else 0.0
                    change_5d = ((current - prev_5d) / prev_5d) * 100 if prev_5d != 0 else 0.0
                    history = series.tail(30).tolist()

                    asset = MacroAsset(
                        symbol=key,
                        price=current,
                        change_pct_1d=change_1d,
                        change_pct_5d=change_5d,
                        price_history=history,
                    )
                    setattr(data, key.lower().replace("10", ""), asset if key != "US10Y" else None)
                    # Manual assignment for each key
                    if key == "DXY":
                        data.dxy = asset
                    elif key == "US10Y":
                        data.us10y = asset
                    elif key == "SPY":
                        data.spy = asset
                    elif key == "VIX":
                        data.vix = asset
                    elif key == "USO":
                        data.uso = asset
                except Exception as e:
                    logger.warning("macro_ticker_failed", ticker=ticker, error=str(e))

        except Exception as e:
            logger.error("macro_fetch_failed", error=str(e))

        return data

    async def run(self) -> None:
        """Main stream loop."""
        logger.info("macro_stream_started")
        while True:
            try:
                macro_data = await asyncio.get_event_loop().run_in_executor(None, self._fetch_macro)
                event = Event(type=EventType.MACRO_UPDATE, data=macro_data, source="macro_stream")
                await self.bus.publish(event)
                logger.debug(
                    "macro_update_published",
                    dxy=macro_data.dxy.price if macro_data.dxy else None,
                    vix=macro_data.vix.price if macro_data.vix else None,
                )
            except asyncio.CancelledError:
                logger.info("macro_stream_stopped")
                break
            except Exception as e:
                logger.error("macro_stream_error", error=str(e))

            await asyncio.sleep(self.config.macro_poll_interval)
