"""Master OSINT coordinator for XAUUSD quant trading system.

Orchestrates:
  - Reddit social ingestion (every 3 min, via run_in_executor)
  - Twitter/X social ingestion (every 3 min, async)
  - Sentiment engine updates
  - Narrative detection
  - GRI computation
  - Fast signal pulse
  - AI summarization (background, 3-min cache)
  - Trade adapter bias
  - Publishes state.osint_data = OsintSnapshot every cycle

Never raises — all failures are swallowed gracefully.
Async polling loop with configurable interval (default 180s / 3 min).
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OsintSnapshot (the single output dataclass written to state.osint_data)
# ---------------------------------------------------------------------------

@dataclass
class OsintSnapshot:
    # Gold Risk Index
    gri_score: float = 50.0
    gri_geo_component: float = 50.0
    gri_monetary_component: float = 50.0
    gri_safe_haven_component: float = 50.0
    gri_retail_component: float = 50.0
    gri_label: str = "moderate"          # low / moderate / elevated / extreme

    # Fast signal pulse
    osint_fast_score: float = 50.0
    osint_fast_delta: float = 0.0
    osint_fast_label: str = "neutral"

    # AI macro summary
    ai_summary: str = ""
    ai_confidence: float = 0.5
    ai_summary_cached_at: float = 0.0

    # Blended composite (final actionable signal)
    blended_composite: float = 50.0
    blended_direction: str = "NEUTRAL"   # BULLISH / BEARISH / NEUTRAL

    # Narratives
    narratives: List[dict] = field(default_factory=list)
    # [{"name": str, "confidence": float, "gold_impact": str, "evidence_count": int}]

    # Social posts (dicts for JSON serialization)
    reddit_posts: List[dict] = field(default_factory=list)
    twitter_posts: List[dict] = field(default_factory=list)
    reddit_available: bool = False
    twitter_available: bool = False

    # Risk flags
    fear_spike: bool = False
    hawkish_dominant: bool = False
    long_multiplier_boost: bool = False  # GRI>85 AND fear_spike AND quant bullish

    # Trade adapter
    trade_bias_rule: str = "DEFAULT"
    trade_size_multiplier: float = 1.0
    trade_direction_bias: str = "NEUTRAL"
    trade_rationale: str = ""

    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# OsintAggregator
# ---------------------------------------------------------------------------

class OsintAggregator:
    """
    Master coordinator that runs an async polling loop.

    Usage:
        agg = OsintAggregator(state, bus)
        asyncio.create_task(agg.run())
    """

    def __init__(self, state, bus=None):
        self._state = state
        self._bus = bus
        self._poll_interval = 180   # seconds (3 min)

        # Lazy-imported engines (avoid heavy init at module load time)
        self._reddit = None
        self._twitter = None
        self._sentiment = None
        self._narrative = None
        self._ai = None
        self._fast = None
        self._gri = None
        self._adapter = None

        self._initialized = False

    def _init_engines(self) -> None:
        """Initialize all engines on first cycle (lazy)."""
        if self._initialized:
            return
        try:
            from osint.reddit_client import RedditClient
            from osint.twitter_client import TwitterClient
            from osint.sentiment_engine import SentimentEngine
            from osint.narrative_detector import NarrativeDetector
            from osint.ai_summarizer import AISummarizer
            from osint.fast_signal_engine import FastSignalEngine
            from osint.gold_risk_index import GoldRiskIndex
            from osint.trade_adapter import TradeAdapter

            self._reddit = RedditClient()
            self._twitter = TwitterClient()
            self._sentiment = SentimentEngine()
            self._narrative = NarrativeDetector()
            self._ai = AISummarizer()
            self._fast = FastSignalEngine()
            self._gri = GoldRiskIndex()
            self._adapter = TradeAdapter()
            self._initialized = True
            logger.info("osint_aggregator_engines_initialized")
        except Exception as e:
            logger.error("osint_aggregator_init_failed: %s", e)

    async def run(self) -> None:
        """
        Main polling loop. Runs indefinitely with self._poll_interval sleep.
        Safe to cancel — exits cleanly on CancelledError.
        """
        logger.info("osint_aggregator_started: poll_interval=%ds", self._poll_interval)
        self._init_engines()

        while True:
            try:
                await self._cycle()
            except asyncio.CancelledError:
                logger.info("osint_aggregator_cancelled")
                raise
            except Exception as e:
                logger.warning("osint_aggregator_cycle_error: %s", e)

            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                logger.info("osint_aggregator_sleep_cancelled")
                raise

    async def _cycle(self) -> None:
        """
        One full OSINT data cycle. Fetches social data, computes all signals,
        writes state.osint_data, and optionally publishes OSINT_UPDATE event.
        """
        t0 = time.monotonic()

        if not self._initialized:
            self._init_engines()
        if not self._initialized:
            return   # engines failed to init — skip

        # --- 1. Fetch social data concurrently ---
        reddit_posts = []
        twitter_posts = []
        try:
            loop = asyncio.get_event_loop()
            reddit_future = loop.run_in_executor(
                None, lambda: self._reddit.fetch_top_posts(25)
            )
            twitter_future = self._twitter.fetch_recent_tweets(25)
            reddit_posts, twitter_posts = await asyncio.gather(
                reddit_future, twitter_future, return_exceptions=True
            )
            if isinstance(reddit_posts, Exception):
                logger.debug("reddit_fetch_exception: %s", reddit_posts)
                reddit_posts = []
            if isinstance(twitter_posts, Exception):
                logger.debug("twitter_fetch_exception: %s", twitter_posts)
                twitter_posts = []
        except Exception as e:
            logger.debug("social_fetch_failed: %s", e)

        # --- 2. Feed texts into sentiment engine + narrative detector ---
        for post in (reddit_posts or []):
            combined = f"{getattr(post, 'title', '')} {getattr(post, 'text', '')}"
            relevance = getattr(post, "engagement_score", 0.5)
            self._sentiment.add_text(combined, source="reddit", relevance=float(relevance))
            self._narrative.add_text(combined)

        for tweet in (twitter_posts or []):
            text = getattr(tweet, "text", "")
            relevance = getattr(tweet, "engagement_score", 0.5)
            self._sentiment.add_text(text, source="twitter", relevance=float(relevance))
            self._narrative.add_text(text)

        # Feed recent quant events into narrative detector
        recent_events = getattr(self._state, "recent_events", [])
        for entry in list(recent_events)[-20:]:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                self._narrative.add_text(str(entry[1]))

        # --- 3. Compute sentiment snapshot ---
        sentiment_snap = self._sentiment.compute()

        # --- 4. Detect narratives ---
        narrative_signals = self._narrative.detect()
        hawkish_index = self._narrative.hawkish_index

        # Convert to list of dicts for JSON serialization
        narratives_dicts = [
            {
                "name": n.name,
                "confidence": n.confidence,
                "gold_impact": n.gold_impact,
                "description": n.description,
                "evidence_count": n.evidence_count,
            }
            for n in narrative_signals
        ]

        # --- 5. Get current quant state ---
        macro_snap = getattr(self._state, "macro_snap", None)
        quant_composite = 50.0
        quant_direction = "NEUTRAL"
        try:
            signal_score = getattr(self._state, "signal_score", None)
            if signal_score is not None:
                raw = getattr(signal_score, "composite", None)
                if raw is not None:
                    import math
                    if not (math.isnan(raw) or math.isinf(raw)):
                        quant_composite = float(raw)
                direction = getattr(signal_score, "direction", None)
                if direction:
                    quant_direction = str(direction)
        except Exception:
            pass

        # --- 6. GRI computation ---
        gri_snap = self._gri.compute(
            narratives=narrative_signals,
            sentiment_snap=sentiment_snap,
            macro_snap=macro_snap,
            recent_events=recent_events,
            quant_composite=quant_composite,
        )

        # --- 7. Fast signal pulse ---
        fast_snap = self._fast.compute(
            sentiment_snap=sentiment_snap,
            narratives=narrative_signals,
            gri_score=gri_snap.score,
        )

        # --- 8. AI summarization (non-blocking, 3-min cache) ---
        context_text = self._ai.build_context(
            narratives=narratives_dicts,
            sentiment_snap=sentiment_snap,
            recent_events=recent_events,
        )
        await self._ai.refresh_if_needed(context_text)
        ai_summary, ai_confidence = self._ai.get_cached_summary()

        # --- 9. Trade adapter bias ---
        biases = self._adapter.compute_bias(
            gri_snap=gri_snap,
            fast_snap=fast_snap,
            narratives=narrative_signals,
            hawkish_index=hawkish_index,
            quant_direction=quant_direction,
            quant_composite=quant_composite,
        )
        bias = biases[0] if biases else None

        # --- 10. Blend composite ---
        blended = self._blend_composite(
            quant=quant_composite,
            gri=gri_snap.score,
            fast=fast_snap.score,
            ai_conf=ai_confidence,
        )
        blended_direction = _score_to_direction(blended)

        # --- 11. Derive risk flags ---
        fear_spike = sentiment_snap.fear_spike
        hawkish_dominant = (
            hawkish_index > 75.0
            and self._narrative.dominant_narrative is not None
            and getattr(self._narrative.dominant_narrative, "name", "") == "USD_DOMINANCE"
        )
        long_multiplier_boost = (
            gri_snap.score > 85.0
            and fear_spike
            and quant_composite >= 55.0
        )

        # --- 12. Serialize social posts to dicts ---
        reddit_dicts = _posts_to_dicts(reddit_posts or [])[:10]
        twitter_dicts = _tweets_to_dicts(twitter_posts or [])[:10]

        # --- 13. Build snapshot ---
        snapshot = OsintSnapshot(
            gri_score=gri_snap.score,
            gri_geo_component=gri_snap.geo_component,
            gri_monetary_component=gri_snap.monetary_component,
            gri_safe_haven_component=gri_snap.safe_haven_component,
            gri_retail_component=gri_snap.retail_component,
            gri_label=gri_snap.label,
            osint_fast_score=fast_snap.score,
            osint_fast_delta=fast_snap.delta,
            osint_fast_label=fast_snap.label,
            ai_summary=ai_summary,
            ai_confidence=ai_confidence,
            ai_summary_cached_at=self._ai.cached_at,
            blended_composite=round(blended, 2),
            blended_direction=blended_direction,
            narratives=narratives_dicts,
            reddit_posts=reddit_dicts,
            twitter_posts=twitter_dicts,
            reddit_available=self._reddit.is_available,
            twitter_available=self._twitter.is_available,
            fear_spike=fear_spike,
            hawkish_dominant=hawkish_dominant,
            long_multiplier_boost=long_multiplier_boost,
            trade_bias_rule=bias.rule_name if bias else "DEFAULT",
            trade_size_multiplier=bias.size_multiplier if bias else 1.0,
            trade_direction_bias=bias.direction_bias if bias else "NEUTRAL",
            trade_rationale=bias.rationale if bias else "",
        )

        # --- 14. Write to state ---
        self._state.osint_data = snapshot

        # --- 15. Publish event (optional) ---
        if self._bus is not None:
            try:
                from quant.core.event_bus import EventType
                await self._bus.publish(EventType.OSINT_UPDATE, snapshot)
            except Exception as e:
                logger.debug("osint_event_publish_failed: %s", e)

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "osint_cycle_complete: gri=%.1f fast=%.1f blended=%.1f "
            "reddit=%d twitter=%d elapsed=%.0fms",
            gri_snap.score,
            fast_snap.score,
            blended,
            len(reddit_posts or []),
            len(twitter_posts or []),
            elapsed,
        )

    def _blend_composite(
        self,
        quant: float,
        gri: float,
        fast: float,
        ai_conf: float,
    ) -> float:
        """
        Blend quant + OSINT signals into final composite.
        Formula: quant(60%) + gri(20%) + fast(10%) + (fast * ai_conf_factor)(10%)

        ai_conf_factor: maps ai_confidence 0..1 → 0.5..1.5 amplifier on fast score
        """
        # Map ai_confidence 0..1 → amplifier 0.5..1.5 (neutral=1.0 at conf=0.5)
        ai_amp = 0.5 + ai_conf
        ai_adjusted_fast = max(0.0, min(100.0, fast * ai_amp / 1.0))
        # Normalize ai_adjusted_fast toward 50 baseline
        ai_component = 50.0 + (ai_adjusted_fast - 50.0) * ai_conf

        blended = (
            quant * 0.60
            + gri * 0.20
            + fast * 0.10
            + ai_component * 0.10
        )
        return round(max(0.0, min(100.0, blended)), 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_to_direction(score: float) -> str:
    """Map 0-100 blended score to direction string."""
    if score >= 58:
        return "BULLISH"
    if score <= 42:
        return "BEARISH"
    return "NEUTRAL"


def _posts_to_dicts(posts: list) -> List[dict]:
    """Convert RedditPost dataclass instances to plain dicts."""
    result = []
    for p in posts:
        try:
            result.append({
                "title": getattr(p, "title", "")[:150],
                "text": getattr(p, "text", "")[:200],
                "subreddit": getattr(p, "subreddit", ""),
                "score": getattr(p, "score", 0),
                "num_comments": getattr(p, "num_comments", 0),
                "url": getattr(p, "url", ""),
                "sentiment_score": getattr(p, "sentiment_score", 0.0),
                "engagement_score": getattr(p, "engagement_score", 0.0),
                "created_utc": getattr(p, "created_utc", 0.0),
                "upvote_ratio": getattr(p, "upvote_ratio", 0.0),
            })
        except Exception:
            pass
    return result


def _tweets_to_dicts(tweets: list) -> List[dict]:
    """Convert Tweet dataclass instances to plain dicts."""
    result = []
    for t in tweets:
        try:
            result.append({
                "text": getattr(t, "text", "")[:280],
                "author": getattr(t, "author", ""),
                "created_at": getattr(t, "created_at", 0.0),
                "like_count": getattr(t, "like_count", 0),
                "retweet_count": getattr(t, "retweet_count", 0),
                "reply_count": getattr(t, "reply_count", 0),
                "url": getattr(t, "url", ""),
                "sentiment_score": getattr(t, "sentiment_score", 0.0),
                "engagement_score": getattr(t, "engagement_score", 0.0),
            })
        except Exception:
            pass
    return result
