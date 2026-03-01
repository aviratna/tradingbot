"""OSINT Sentiment Engine for XAUUSD.

In-memory VADER sentiment analysis with:
- Fear/optimism dimension scoring
- Exponential decay weighting (1-hour half-life)
- Source-aware aggregation (reddit / twitter / news)
- Sub-5ms compute time on 150 items

Never imports heavy ML models. Uses only VADER (already in requirements).
"""

import math
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Fear-signaling words (presence increases fear_index)
FEAR_WORDS = {
    "crash", "collapse", "war", "panic", "crisis", "sanctions", "hyperinflation",
    "recession", "default", "conflict", "attack", "explosion", "meltdown",
    "plunge", "tumble", "spiral", "fear", "shock", "chaos", "turmoil",
    "catastrophe", "disaster", "implosion", "selloff", "dump", "bloodbath",
    "stagflation", "depression", "bankruptcy", "contagion", "bubble burst",
}

# Optimism-signaling words
OPTIMISM_WORDS = {
    "rally", "surge", "breakout", "bullish", "strong", "recovery", "growth",
    "momentum", "record", "all-time high", "outperform", "upside", "boom",
    "expansion", "robust", "resilient", "stabilize", "rebound", "jump",
    "soar", "climb", "advance", "gain", "rise", "positive", "optimistic",
    "confidence", "opportunity", "uptrend",
}


@dataclass
class SentimentReading:
    polarity: float       # VADER compound -1..1
    fear_score: float     # 0..1 (proportion of fear words)
    optimism_score: float # 0..1 (proportion of optimism words)
    relevance: float      # 0..1 (engagement-based weight)
    source: str           # "reddit" | "twitter" | "news" | "other"
    timestamp: float = field(default_factory=time.time)


@dataclass
class SentimentEngineSnapshot:
    composite_polarity: float    # -1..1 (recency + relevance weighted)
    fear_index: float            # 0..1
    optimism_index: float        # 0..1
    fear_spike: bool             # True when fear_index > 0.65
    normalized_score: float      # 0..100 (50 = neutral)
    source_counts: dict          # {"reddit": N, "twitter": N, "news": N}
    item_count: int              # total items in buffer
    timestamp: float = field(default_factory=time.time)


def _fear_score(text: str) -> float:
    """Compute 0..1 fear word density."""
    words = text.lower().split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if any(fw in w for fw in FEAR_WORDS))
    return min(1.0, hits / max(len(words), 1) * 10)   # scale so 1/10 words = 1.0


def _optimism_score(text: str) -> float:
    """Compute 0..1 optimism word density."""
    words = text.lower().split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if any(ow in w for ow in OPTIMISM_WORDS))
    return min(1.0, hits / max(len(words), 1) * 10)


def _vader_compound(text: str) -> float:
    """VADER compound score -1..1. Returns 0.0 on failure (lazy init)."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        # Lazy module-level singleton
        if not hasattr(_vader_compound, "_sia"):
            _vader_compound._sia = SentimentIntensityAnalyzer()
        return _vader_compound._sia.polarity_scores(text)["compound"]
    except Exception:
        return 0.0


class SentimentEngine:
    """
    Maintains a rolling deque of sentiment readings with exponential decay.

    Compute time: <5ms on 150 items (pure Python, no numpy).
    Thread-safe: designed for single-threaded async access.
    """

    MAX_ITEMS = 150
    HALF_LIFE_SECONDS = 3600       # 1-hour half-life
    FEAR_SPIKE_THRESHOLD = 0.55    # fear_index above this → fear_spike=True

    def __init__(self):
        self._items: deque = deque(maxlen=self.MAX_ITEMS)

    def add_text(self, text: str, source: str = "other", relevance: float = 1.0) -> None:
        """
        Analyze text and store reading. Fast path: returns immediately on empty text.
        relevance: caller-provided engagement weight (0..1 normalized).
        """
        if not text or not text.strip():
            return
        try:
            polarity = _vader_compound(text[:1000])   # cap text for speed
            fear = _fear_score(text[:500])
            optimism = _optimism_score(text[:500])
            self._items.append(SentimentReading(
                polarity=polarity,
                fear_score=fear,
                optimism_score=optimism,
                relevance=max(0.0, min(1.0, relevance)),
                source=source,
            ))
        except Exception as e:
            logger.debug(f"sentiment_add_text_failed: {e}")

    def compute(self) -> SentimentEngineSnapshot:
        """
        Compute aggregate sentiment snapshot with exponential decay weighting.
        Returns neutral snapshot (50.0) if no items.
        """
        if not self._items:
            return SentimentEngineSnapshot(
                composite_polarity=0.0,
                fear_index=0.0,
                optimism_index=0.0,
                fear_spike=False,
                normalized_score=50.0,
                source_counts={"reddit": 0, "twitter": 0, "news": 0},
                item_count=0,
            )

        now = time.time()
        decay_k = math.log(2) / self.HALF_LIFE_SECONDS

        total_weight = 0.0
        weighted_polarity = 0.0
        weighted_fear = 0.0
        weighted_optimism = 0.0
        source_counts = {"reddit": 0, "twitter": 0, "news": 0}

        for item in self._items:
            age = now - item.timestamp
            decay = math.exp(-decay_k * age)
            weight = decay * (0.3 + 0.7 * item.relevance)   # min 30% weight floor

            total_weight += weight
            weighted_polarity += item.polarity * weight
            weighted_fear += item.fear_score * weight
            weighted_optimism += item.optimism_score * weight

            src = item.source
            if src in source_counts:
                source_counts[src] += 1
            else:
                source_counts[src] = source_counts.get(src, 0) + 1

        if total_weight == 0:
            total_weight = 1.0

        composite_polarity = weighted_polarity / total_weight
        fear_index = min(1.0, weighted_fear / total_weight)
        optimism_index = min(1.0, weighted_optimism / total_weight)

        # Normalize polarity -1..1 → 0..100 (50 = neutral)
        normalized_score = 50.0 + composite_polarity * 50.0
        normalized_score = max(0.0, min(100.0, normalized_score))

        return SentimentEngineSnapshot(
            composite_polarity=round(composite_polarity, 4),
            fear_index=round(fear_index, 4),
            optimism_index=round(optimism_index, 4),
            fear_spike=fear_index > self.FEAR_SPIKE_THRESHOLD,
            normalized_score=round(normalized_score, 2),
            source_counts=dict(source_counts),
            item_count=len(self._items),
        )

    def clear(self) -> None:
        """Clear all stored items."""
        self._items.clear()
