"""Polymarket prediction market stream, tracking probability changes."""

import asyncio
import sys
from pathlib import Path
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.event_bus import Event, EventType, get_event_bus
from quant.core.config import get_config
from quant.core.logger import get_logger
from app.data.polymarket import PolymarketFetcher

logger = get_logger(__name__)


@dataclass
class PolyUpdate:
    metals_events: List[dict]       # metals-relevant markets
    geo_events: List[dict]          # geopolitical risk markets
    trending: List[dict]            # top trending markets
    risk_off_index: float           # 0.0 to 1.0 aggregate risk-off signal
    overall_bias: str               # bullish / bearish / neutral
    prob_deltas: Dict[str, float]   # {market_id: prob_change since last poll}
    timestamp: float = field(default_factory=time.time)


class PolymarketStream:
    """Polls Polymarket Gamma API and tracks probability changes."""

    def __init__(self):
        self.config = get_config()
        self.bus = get_event_bus()
        self._fetcher = PolymarketFetcher()
        self._prev_probs: Dict[str, float] = {}

    def _compute_risk_off_index(self, geo_events: List[dict]) -> float:
        """Compute a 0-1 risk-off index from geopolitical prediction markets."""
        if not geo_events:
            return 0.3  # baseline
        # High yes-probability on war/conflict = risk-off signal
        risk_signals = []
        for evt in geo_events:
            yes_prob = evt.get("yes_probability", 0.5)
            relevance = evt.get("relevance_score", 0.5)
            risk_signals.append(yes_prob * relevance)

        if not risk_signals:
            return 0.3
        return min(sum(risk_signals) / len(risk_signals) * 1.5, 1.0)

    def _compute_prob_deltas(self, current_events: List[dict]) -> Dict[str, float]:
        """Compute probability changes since last poll."""
        deltas = {}
        for evt in current_events:
            market_id = str(evt.get("id", ""))
            current_prob = evt.get("yes_probability", 0.5)
            if market_id in self._prev_probs:
                deltas[market_id] = current_prob - self._prev_probs[market_id]
            self._prev_probs[market_id] = current_prob
        return deltas

    def _fetch_all(self) -> Optional[PolyUpdate]:
        """Fetch all Polymarket data synchronously."""
        try:
            # Use cached fetcher methods
            metals_intel = self._fetcher.get_metals_relevant_events()
            geo = self._fetcher.get_geopolitical_events()
            trending = self._fetcher.get_trending_events()

            metals_list = [
                {
                    "id": e.id,
                    "title": e.title,
                    "yes_probability": e.yes_probability,
                    "volume": e.volume,
                    "relevance_score": e.relevance_score,
                    "price_signal": e.price_signal,
                }
                for e in metals_intel
            ]
            geo_list = [
                {
                    "id": e.id,
                    "title": e.title,
                    "yes_probability": e.yes_probability,
                    "volume": e.volume,
                    "relevance_score": e.relevance_score,
                }
                for e in geo
            ]
            trending_list = [
                {
                    "id": e.id,
                    "title": e.title,
                    "yes_probability": e.yes_probability,
                    "volume": e.volume,
                    "price_signal": e.price_signal,
                }
                for e in trending[:10]
            ]

            all_events = metals_list + geo_list
            prob_deltas = self._compute_prob_deltas(all_events)
            risk_off = self._compute_risk_off_index(geo_list)

            # Determine overall bias from metals markets
            bullish_count = sum(1 for e in metals_list if e.get("price_signal") == "bullish")
            bearish_count = sum(1 for e in metals_list if e.get("price_signal") == "bearish")
            if bullish_count > bearish_count:
                bias = "bullish"
            elif bearish_count > bullish_count:
                bias = "bearish"
            else:
                bias = "neutral"

            return PolyUpdate(
                metals_events=metals_list,
                geo_events=geo_list,
                trending=trending_list,
                risk_off_index=risk_off,
                overall_bias=bias,
                prob_deltas=prob_deltas,
            )
        except Exception as e:
            logger.error("polymarket_fetch_failed", error=str(e))
            return None

    async def run(self) -> None:
        """Main stream loop."""
        logger.info("polymarket_stream_started")
        while True:
            try:
                update = await asyncio.get_event_loop().run_in_executor(None, self._fetch_all)
                if update:
                    event = Event(type=EventType.POLY_UPDATE, data=update, source="polymarket_stream")
                    await self.bus.publish(event)
                    logger.debug(
                        "polymarket_published",
                        metals_count=len(update.metals_events),
                        bias=update.overall_bias,
                        risk_off=round(update.risk_off_index, 3),
                    )
            except asyncio.CancelledError:
                logger.info("polymarket_stream_stopped")
                break
            except Exception as e:
                logger.error("polymarket_stream_error", error=str(e))

            await asyncio.sleep(self.config.polymarket_poll_interval)
