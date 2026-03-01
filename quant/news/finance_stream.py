"""Finance news stream via RSS feeds with sentiment scoring."""

import asyncio
import sys
from pathlib import Path
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Set, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.event_bus import Event, EventType, get_event_bus
from quant.core.config import get_config
from quant.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NewsItem:
    title: str
    summary: str
    url: str
    source: str
    sentiment_score: float      # -1.0 to 1.0
    sentiment_label: str        # positive/negative/neutral
    relevance_score: float      # 0.0 to 1.0 (gold relevance)
    timestamp: float = field(default_factory=time.time)
    item_id: str = ""           # hash for dedup


class FinanceNewsStream:
    """Polls finance RSS feeds and scores sentiment."""

    def __init__(self):
        self.config = get_config()
        self.bus = get_event_bus()
        self._seen_ids: Set[str] = set()
        self._seen_ids_max = 1000

    def _score_relevance(self, text: str) -> float:
        """Score how relevant this news item is to gold/metals."""
        text_lower = text.lower()
        score = 0.0
        high_value_keywords = ["gold", "xau", "silver", "bullion", "precious metal", "xaut"]
        med_value_keywords = ["inflation", "fed ", "federal reserve", "dollar", "dxy", "treasury", "yield"]
        low_value_keywords = ["war", "conflict", "geopolit", "safe haven", "risk off", "recession", "stagflation"]

        for kw in high_value_keywords:
            if kw in text_lower:
                score += 0.3
        for kw in med_value_keywords:
            if kw in text_lower:
                score += 0.15
        for kw in low_value_keywords:
            if kw in text_lower:
                score += 0.08

        return min(score, 1.0)

    def _analyze_sentiment(self, text: str) -> tuple:
        """Run VADER sentiment analysis."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            compound = scores["compound"]
            if compound >= 0.05:
                label = "positive"
            elif compound <= -0.05:
                label = "negative"
            else:
                label = "neutral"
            return compound, label
        except Exception:
            return 0.0, "neutral"

    def _fetch_feeds(self) -> List[NewsItem]:
        """Fetch and parse all RSS feeds."""
        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser_not_installed")
            return []

        items = []
        for feed_url in self.config.finance_rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:  # max 10 per feed
                    title = getattr(entry, "title", "")
                    summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
                    url = getattr(entry, "link", "")
                    published = getattr(entry, "published", "")

                    item_id = hashlib.md5(f"{title}{url}".encode()).hexdigest()
                    if item_id in self._seen_ids:
                        continue

                    text = f"{title} {summary}"
                    relevance = self._score_relevance(text)
                    if relevance < 0.08:
                        continue  # skip irrelevant items

                    sentiment, label = self._analyze_sentiment(text)
                    source = feed.feed.get("title", feed_url.split("/")[2])

                    item = NewsItem(
                        title=title[:200],
                        summary=summary[:500],
                        url=url,
                        source=source,
                        sentiment_score=sentiment,
                        sentiment_label=label,
                        relevance_score=relevance,
                        item_id=item_id,
                    )
                    items.append(item)

                    self._seen_ids.add(item_id)
                    if len(self._seen_ids) > self._seen_ids_max:
                        # Prune oldest
                        self._seen_ids = set(list(self._seen_ids)[-500:])

            except Exception as e:
                logger.warning("feed_parse_failed", url=feed_url, error=str(e))

        return items

    async def run(self) -> None:
        """Main stream loop."""
        logger.info("finance_stream_started")
        while True:
            try:
                items = await asyncio.get_event_loop().run_in_executor(None, self._fetch_feeds)
                for item in items:
                    event = Event(type=EventType.NEWS_ITEM, data=item, source="finance_stream")
                    await self.bus.publish(event)
                if items:
                    logger.debug("finance_news_published", count=len(items))
            except asyncio.CancelledError:
                logger.info("finance_stream_stopped")
                break
            except Exception as e:
                logger.error("finance_stream_error", error=str(e))

            await asyncio.sleep(self.config.news_poll_interval)
