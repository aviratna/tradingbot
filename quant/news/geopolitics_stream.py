"""Geopolitical news stream via RSS feeds with severity scoring."""

import asyncio
import sys
from pathlib import Path
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.event_bus import Event, EventType, get_event_bus
from quant.core.config import get_config
from quant.core.logger import get_logger

logger = get_logger(__name__)

GEO_HIGH_SEVERITY = [
    "war", "attack", "missile", "nuclear", "invasion", "conflict", "military",
    "bomb", "sanction", "escalat", "coup", "terror", "explosion"
]
GEO_MED_SEVERITY = [
    "tension", "crisis", "protest", "threat", "warning", "sanction",
    "troops", "border", "ceasefire", "diplomatic", "clash"
]
GOLD_GEO_KEYWORDS = [
    "gold", "oil", "energy", "dollar", "fed", "inflation", "russia", "ukraine",
    "china", "iran", "israel", "middle east", "nato", "us-china", "taiwan",
    "commodity", "safe haven", "precious", "reserve", "central bank"
]


@dataclass
class GeoEvent:
    title: str
    summary: str
    url: str
    source: str
    severity: float         # 0.0 (low) to 1.0 (critical)
    severity_label: str     # low / medium / high / critical
    gold_impact: str        # bullish / bearish / neutral
    sentiment_score: float
    timestamp: float = field(default_factory=time.time)
    item_id: str = ""


class GeopoliticsStream:
    """Polls geopolitical RSS feeds and scores severity/gold impact."""

    def __init__(self):
        self.config = get_config()
        self.bus = get_event_bus()
        self._seen_ids: Set[str] = set()

    def _score_severity(self, text: str) -> tuple:
        """Score geopolitical severity."""
        text_lower = text.lower()
        score = 0.0
        for kw in GEO_HIGH_SEVERITY:
            if kw in text_lower:
                score += 0.3
        for kw in GEO_MED_SEVERITY:
            if kw in text_lower:
                score += 0.15
        score = min(score, 1.0)

        if score >= 0.7:
            label = "critical"
        elif score >= 0.4:
            label = "high"
        elif score >= 0.15:
            label = "medium"
        else:
            label = "low"

        return score, label

    def _score_gold_impact(self, text: str, severity: float) -> str:
        """Determine if event is bullish/bearish/neutral for gold."""
        text_lower = text.lower()
        # High severity geo events are generally bullish for gold (safe haven)
        if severity >= 0.5:
            return "bullish"
        # Dollar strength is bearish for gold
        bearish_cues = ["strong dollar", "rate hike", "hawkish", "taper", "sell-off"]
        for cue in bearish_cues:
            if cue in text_lower:
                return "bearish"
        # Gold-specific mentions with positive context
        if any(kw in text_lower for kw in ["safe haven", "gold rally", "gold surge", "precious metal"]):
            return "bullish"
        return "neutral"

    def _analyze_sentiment(self, text: str) -> float:
        """Run VADER sentiment."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            return analyzer.polarity_scores(text)["compound"]
        except Exception:
            return 0.0

    def _fetch_feeds(self) -> List[GeoEvent]:
        """Fetch geopolitical RSS feeds."""
        try:
            import feedparser
        except ImportError:
            return []

        events = []
        for feed_url in self.config.geopolitics_rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:
                    title = getattr(entry, "title", "")
                    summary = getattr(entry, "summary", "") or ""
                    url = getattr(entry, "link", "")
                    text = f"{title} {summary}"

                    # Filter for geo/gold relevance
                    text_lower = text.lower()
                    if not any(kw in text_lower for kw in GEO_HIGH_SEVERITY + GEO_MED_SEVERITY + GOLD_GEO_KEYWORDS):
                        continue

                    item_id = hashlib.md5(f"{title}{url}".encode()).hexdigest()
                    if item_id in self._seen_ids:
                        continue

                    severity, sev_label = self._score_severity(text)
                    gold_impact = self._score_gold_impact(text, severity)
                    sentiment = self._analyze_sentiment(text)
                    source = feed.feed.get("title", feed_url.split("/")[2])

                    event = GeoEvent(
                        title=title[:200],
                        summary=summary[:500],
                        url=url,
                        source=source,
                        severity=severity,
                        severity_label=sev_label,
                        gold_impact=gold_impact,
                        sentiment_score=sentiment,
                        item_id=item_id,
                    )
                    events.append(event)
                    self._seen_ids.add(item_id)
                    if len(self._seen_ids) > 1000:
                        self._seen_ids = set(list(self._seen_ids)[-500:])

            except Exception as e:
                logger.warning("geo_feed_failed", url=feed_url, error=str(e))

        return events

    async def run(self) -> None:
        """Main stream loop."""
        logger.info("geopolitics_stream_started")
        while True:
            try:
                events = await asyncio.get_event_loop().run_in_executor(None, self._fetch_feeds)
                for evt in events:
                    event = Event(type=EventType.GEO_EVENT, data=evt, source="geo_stream")
                    await self.bus.publish(event)
                if events:
                    logger.debug("geo_events_published", count=len(events))
            except asyncio.CancelledError:
                logger.info("geo_stream_stopped")
                break
            except Exception as e:
                logger.error("geo_stream_error", error=str(e))

            await asyncio.sleep(self.config.news_poll_interval)
