"""Fast in-memory OSINT signal pulse for XAUUSD.

Produces a 0-100 score in <100ms (pure Python, no I/O, no ML models).

Formula:
  sentiment_component (40%) → from SentimentEngineSnapshot
  narrative_bias     (35%) → net bullish/bearish from NarrativeSignals
  gri_normalized     (25%) → GRI score already in 0-100

Output: FastSignalSnapshot with score, delta vs previous, and label.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Weight constants
_W_SENTIMENT = 0.40
_W_NARRATIVE = 0.35
_W_GRI = 0.25


@dataclass
class FastSignalSnapshot:
    score: float         # 0..100 (50 = neutral)
    delta: float         # change vs previous snapshot (can be negative)
    label: str           # "strongly_bullish" | "bullish" | "neutral" | "bearish" | "strongly_bearish"
    sentiment_component: float   # 0..100 contribution from sentiment
    narrative_component: float   # 0..100 contribution from narratives
    gri_component: float         # 0..100 contribution from GRI
    timestamp: float = field(default_factory=time.time)


def _score_to_label(score: float) -> str:
    """Map 0-100 score to descriptive label."""
    if score >= 70:
        return "strongly_bullish"
    if score >= 58:
        return "bullish"
    if score <= 30:
        return "strongly_bearish"
    if score <= 42:
        return "bearish"
    return "neutral"


def _compute_narrative_score(narratives: List) -> float:
    """
    Compute 0-100 score from narrative signals.
    Bullish narratives push above 50, bearish below 50.
    """
    if not narratives:
        return 50.0

    bullish_weight = 0.0
    bearish_weight = 0.0

    for n in narratives:
        # Handle both dataclass and dict
        if isinstance(n, dict):
            confidence = n.get("confidence", 0.0)
            gold_impact = n.get("gold_impact", "neutral")
        else:
            confidence = getattr(n, "confidence", 0.0)
            gold_impact = getattr(n, "gold_impact", "neutral")

        if gold_impact == "bullish":
            bullish_weight += confidence
        elif gold_impact == "bearish":
            bearish_weight += confidence

    total = bullish_weight + bearish_weight
    if total == 0:
        return 50.0

    # Normalize to 0-100 centered at 50
    net = (bullish_weight - bearish_weight) / total
    return round(50.0 + net * 50.0, 2)


class FastSignalEngine:
    """
    Computes a lightweight OSINT signal pulse from already-computed snapshots.
    No I/O, no blocking calls — all inputs are pre-computed in-memory objects.

    Maintains last score for delta computation.
    """

    def __init__(self):
        self._last_score: Optional[float] = None

    def compute(
        self,
        sentiment_snap,
        narratives: List,
        gri_score: float = 50.0,
    ) -> FastSignalSnapshot:
        """
        Compute fast signal score.

        Args:
            sentiment_snap: SentimentEngineSnapshot (or None)
            narratives: List of NarrativeSignal or dicts
            gri_score: Gold Risk Index score 0-100 (already computed)

        Returns:
            FastSignalSnapshot with score, delta, label
        """
        try:
            # --- Sentiment component (0-100) ---
            if sentiment_snap is not None:
                sent_score = getattr(sentiment_snap, "normalized_score", 50.0)
            else:
                sent_score = 50.0

            # --- Narrative component (0-100) ---
            narrative_score = _compute_narrative_score(narratives or [])

            # --- GRI component (already 0-100, passed directly) ---
            gri_clamped = max(0.0, min(100.0, float(gri_score)))

            # --- Weighted composite ---
            composite = (
                sent_score * _W_SENTIMENT
                + narrative_score * _W_NARRATIVE
                + gri_clamped * _W_GRI
            )
            composite = round(max(0.0, min(100.0, composite)), 2)

            # --- Delta vs previous ---
            delta = 0.0
            if self._last_score is not None:
                delta = round(composite - self._last_score, 2)
            self._last_score = composite

            return FastSignalSnapshot(
                score=composite,
                delta=delta,
                label=_score_to_label(composite),
                sentiment_component=round(sent_score, 2),
                narrative_component=round(narrative_score, 2),
                gri_component=round(gri_clamped, 2),
            )

        except Exception as e:
            logger.warning("fast_signal_compute_failed: %s", e)
            return FastSignalSnapshot(
                score=50.0,
                delta=0.0,
                label="neutral",
                sentiment_component=50.0,
                narrative_component=50.0,
                gri_component=50.0,
            )
