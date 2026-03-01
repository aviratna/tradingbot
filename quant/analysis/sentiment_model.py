"""Aggregate news + reddit sentiment with recency weighting → SentimentSnapshot (0-100)."""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Deque
from collections import deque
import time
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.logger import get_logger

logger = get_logger(__name__)

# Half-life of 2 hours for recency weighting (in seconds)
HALF_LIFE_SECONDS = 7200


@dataclass
class SentimentSnapshot:
    # Scores
    news_sentiment: float = 0.0        # -1.0 to 1.0
    reddit_sentiment: float = 0.0      # -1.0 to 1.0
    geo_sentiment: float = 0.0         # -1.0 to 1.0

    # Normalized 0-100 composite
    score: float = 50.0

    # Counts
    news_item_count: int = 0
    reddit_item_count: int = 0
    geo_item_count: int = 0

    # Dominant sentiment
    overall_sentiment: str = "neutral"  # bullish / bearish / neutral
    top_keywords: List[str] = field(default_factory=list)

    timestamp: float = field(default_factory=time.time)


class SentimentModel:
    """Aggregates incoming news, geo, and social sentiment items."""

    MAX_ITEMS = 200

    def __init__(self):
        # Stores (timestamp, sentiment_score, relevance, source_type)
        self._news_items: Deque[tuple] = deque(maxlen=self.MAX_ITEMS)
        self._reddit_items: Deque[tuple] = deque(maxlen=self.MAX_ITEMS)
        self._geo_items: Deque[tuple] = deque(maxlen=self.MAX_ITEMS)

    def add_news(self, item) -> None:
        """Add a NewsItem to the sentiment model."""
        self._news_items.append((item.timestamp, item.sentiment_score, item.relevance_score))

    def add_reddit(self, item) -> None:
        """Add a SocialItem to the sentiment model."""
        self._reddit_items.append((item.timestamp, item.sentiment_score, item.relevance_score))

    def add_geo(self, item) -> None:
        """Add a GeoEvent (severity → sentiment direction for gold)."""
        # High severity geo events → bullish for gold (safe haven)
        sentiment = item.sentiment_score
        # Adjust: war/conflict events bias gold bullish regardless of headline sentiment
        if item.gold_impact == "bullish":
            sentiment = max(sentiment, 0.3)
        elif item.gold_impact == "bearish":
            sentiment = min(sentiment, -0.1)
        self._geo_items.append((item.timestamp, sentiment, item.severity))

    def _weighted_sentiment(self, items: Deque[tuple]) -> tuple:
        """Compute recency-weighted average sentiment."""
        if not items:
            return 0.0, 0

        now = time.time()
        total_weight = 0.0
        weighted_sum = 0.0

        for ts, sentiment, relevance in items:
            age = now - ts
            # Exponential decay by half-life
            weight = relevance * math.exp(-age * math.log(2) / HALF_LIFE_SECONDS)
            weighted_sum += sentiment * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0, len(items)

        return weighted_sum / total_weight, len(items)

    def compute(self) -> SentimentSnapshot:
        """Compute current sentiment snapshot."""
        news_sent, news_count = self._weighted_sentiment(self._news_items)
        reddit_sent, reddit_count = self._weighted_sentiment(self._reddit_items)
        geo_sent, geo_count = self._weighted_sentiment(self._geo_items)

        # Weighted composite: news 40%, reddit 30%, geo 30%
        weights = [0.40, 0.30, 0.30]
        values = [news_sent, reddit_sent, geo_sent]

        # Filter out channels with no data
        active = [(w, v) for w, v in zip(weights, values)
                  if abs(v) > 0 or (w == 0.40 and news_count > 0)]

        if active:
            total_w = sum(w for w, _ in active)
            composite_raw = sum(w * v for w, v in active) / total_w if total_w > 0 else 0.0
        else:
            composite_raw = 0.0

        # Map -1..1 → 0..100
        score = (composite_raw + 1.0) * 50.0
        score = max(0.0, min(100.0, score))

        # Overall label
        if score >= 60:
            overall = "bullish"
        elif score <= 40:
            overall = "bearish"
        else:
            overall = "neutral"

        return SentimentSnapshot(
            news_sentiment=round(news_sent, 4),
            reddit_sentiment=round(reddit_sent, 4),
            geo_sentiment=round(geo_sent, 4),
            score=round(score, 2),
            news_item_count=news_count,
            reddit_item_count=reddit_count,
            geo_item_count=geo_count,
            overall_sentiment=overall,
        )
