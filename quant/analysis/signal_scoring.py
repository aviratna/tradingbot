"""Composite signal scoring: tech(30%) + macro(30%) + sentiment(20%) + polymarket(20%)."""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.config import get_config
from quant.core.logger import get_logger

logger = get_logger(__name__)


class SignalDirection(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


SIGNAL_COLORS = {
    SignalDirection.STRONG_BUY: "bright_green",
    SignalDirection.BUY: "green",
    SignalDirection.NEUTRAL: "yellow",
    SignalDirection.SELL: "red",
    SignalDirection.STRONG_SELL: "bright_red",
}

SIGNAL_EMOJIS = {
    SignalDirection.STRONG_BUY: "🚀",
    SignalDirection.BUY: "📈",
    SignalDirection.NEUTRAL: "⚖️",
    SignalDirection.SELL: "📉",
    SignalDirection.STRONG_SELL: "🔻",
}


@dataclass
class SignalScore:
    # Component scores (0-100 each)
    tech_score: float
    macro_score: float
    sentiment_score: float
    polymarket_score: float

    # Weights
    tech_weight: float
    macro_weight: float
    sentiment_weight: float
    polymarket_weight: float

    # Composite
    composite: float
    direction: SignalDirection
    color: str
    emoji: str

    # Confidence (based on data completeness)
    confidence: float

    # Regime adjustment applied
    regime_adjusted: bool = False
    regime_name: str = "NORMAL"

    timestamp: float = field(default_factory=time.time)


def compute_polymarket_score(poly_data) -> float:
    """Convert Polymarket data to 0-100 bullish/bearish score for gold."""
    if not poly_data:
        return 50.0

    # Start neutral
    score = 50.0

    # Bias adjustment
    bias = getattr(poly_data, "overall_bias", "neutral")
    if bias == "bullish":
        score += 15
    elif bias == "bearish":
        score -= 15

    # Risk-off index (higher risk-off = bullish for gold)
    risk_off = getattr(poly_data, "risk_off_index", 0.3)
    # Map 0..1 → adjustment: 0.3 baseline = 0, above = bullish, below = bearish
    score += (risk_off - 0.3) * 40

    return max(0.0, min(100.0, score))


def score_signal(
    tech_snap,
    macro_snap,
    sentiment_snap,
    poly_data,
    regime_snap=None,
) -> SignalScore:
    """Compute composite 0-100 signal score."""
    cfg = get_config()

    tech_score = tech_snap.score if tech_snap else 50.0
    macro_score = macro_snap.score if macro_snap else 50.0
    sentiment_score = sentiment_snap.score if sentiment_snap else 50.0
    polymarket_score = compute_polymarket_score(poly_data)

    # Track data completeness for confidence
    data_sources = sum([
        tech_snap is not None,
        macro_snap is not None,
        sentiment_snap is not None,
        poly_data is not None,
    ])
    confidence = data_sources / 4.0

    # Weighted composite
    composite = (
        tech_score * cfg.tech_weight +
        macro_score * cfg.macro_weight +
        sentiment_score * cfg.sentiment_weight +
        polymarket_score * cfg.polymarket_weight
    )

    # Regime adjustment
    regime_adjusted = False
    regime_name = "NORMAL"
    if regime_snap:
        regime_name = regime_snap.regime.value
        # Boost/suppress based on regime gold bias
        bias = regime_snap.gold_bias
        adj = regime_snap.confidence * 8  # max 8-point regime adjustment
        if bias == "bullish":
            composite = min(composite + adj, 100)
            regime_adjusted = True
        elif bias == "bearish":
            composite = max(composite - adj, 0)
            regime_adjusted = True

    composite = round(float(composite), 2)

    # Determine direction
    if composite >= 70:
        direction = SignalDirection.STRONG_BUY
    elif composite >= 55:
        direction = SignalDirection.BUY
    elif composite >= 45:
        direction = SignalDirection.NEUTRAL
    elif composite >= 30:
        direction = SignalDirection.SELL
    else:
        direction = SignalDirection.STRONG_SELL

    logger.debug(
        "signal_scored",
        composite=composite,
        direction=direction.value,
        tech=tech_score,
        macro=macro_score,
        sentiment=sentiment_score,
        polymarket=polymarket_score,
    )

    return SignalScore(
        tech_score=round(tech_score, 2),
        macro_score=round(macro_score, 2),
        sentiment_score=round(sentiment_score, 2),
        polymarket_score=round(polymarket_score, 2),
        tech_weight=cfg.tech_weight,
        macro_weight=cfg.macro_weight,
        sentiment_weight=cfg.sentiment_weight,
        polymarket_weight=cfg.polymarket_weight,
        composite=composite,
        direction=direction,
        color=SIGNAL_COLORS[direction],
        emoji=SIGNAL_EMOJIS[direction],
        confidence=confidence,
        regime_adjusted=regime_adjusted,
        regime_name=regime_name,
    )
