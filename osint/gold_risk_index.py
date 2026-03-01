"""Institutional Gold Risk Index (GRI) for XAUUSD.

GRI = geo(0.35) + monetary(0.25) + safe_haven(0.25) + retail(0.15)

Each component is scored 0-100:
  - geo:        geopolitical tensions from SAFE_HAVEN_BID + RECESSION_FEAR narratives
  - monetary:   Fed policy risk from FED_DOVISH vs FED_HAWKISH + inflation narratives
  - safe_haven: safe-haven demand from narratives + fear spike + quant macro
  - retail:     retail/social conviction from INFLATION_FEAR + CENTRAL_BANK_BUYING + sentiment

GRI > 70 → elevated gold demand environment
GRI > 85 → extreme risk conditions (long multiplier trigger zone)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Component weights (must sum to 1.0)
_W_GEO = 0.35
_W_MONETARY = 0.25
_W_SAFE_HAVEN = 0.25
_W_RETAIL = 0.15


@dataclass
class GRISnapshot:
    score: float                  # 0..100 composite
    geo_component: float          # 0..100
    monetary_component: float     # 0..100
    safe_haven_component: float   # 0..100
    retail_component: float       # 0..100
    label: str                    # "low" | "moderate" | "elevated" | "extreme"
    timestamp: float = field(default_factory=time.time)


def _score_to_gri_label(score: float) -> str:
    if score >= 85:
        return "extreme"
    if score >= 70:
        return "elevated"
    if score >= 45:
        return "moderate"
    return "low"


def _get_narrative_confidence(narratives: list, name: str) -> float:
    """Extract confidence for a named narrative from list of NarrativeSignal or dicts."""
    for n in narratives:
        if isinstance(n, dict):
            if n.get("name") == name:
                return float(n.get("confidence", 0.0))
        else:
            if getattr(n, "name", None) == name:
                return float(getattr(n, "confidence", 0.0))
    return 0.0


def _geo_score(narratives: list, recent_events: list) -> float:
    """
    Geopolitical risk score 0-100.
    Primary: SAFE_HAVEN_BID narrative confidence (most direct geo signal).
    Secondary: RECESSION_FEAR (macro fear correlates with geo instability).
    Tertiary: Recent events containing geo keywords.
    """
    safe_haven_conf = _get_narrative_confidence(narratives, "SAFE_HAVEN_BID")
    recession_conf = _get_narrative_confidence(narratives, "RECESSION_FEAR")

    # Geo keywords in recent events (last 10)
    geo_keywords = {
        "war", "conflict", "attack", "sanction", "military",
        "nato", "ukraine", "iran", "nuclear", "explosion", "terror",
    }
    recent_geo_hits = 0
    for entry in list(recent_events)[-10:]:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            msg = str(entry[1]).lower()
        else:
            msg = str(entry).lower()
        if any(kw in msg for kw in geo_keywords):
            recent_geo_hits += 1

    event_boost = min(30.0, recent_geo_hits * 8.0)

    raw = (safe_haven_conf * 60.0) + (recession_conf * 20.0) + event_boost
    return round(min(100.0, raw), 2)


def _monetary_score(narratives: list, macro_snap) -> float:
    """
    Monetary policy risk score 0-100 (for gold).
    Dovish Fed + high inflation = high monetary gold support.
    Hawkish Fed = suppresses monetary component.
    """
    dovish_conf = _get_narrative_confidence(narratives, "FED_DOVISH")
    hawkish_conf = _get_narrative_confidence(narratives, "FED_HAWKISH")
    inflation_conf = _get_narrative_confidence(narratives, "INFLATION_FEAR")

    # Net Fed stance: +1 = fully dovish, -1 = fully hawkish
    fed_total = dovish_conf + hawkish_conf
    if fed_total > 0:
        net_dove = (dovish_conf - hawkish_conf) / fed_total  # -1..1
    else:
        net_dove = 0.0

    # Macro snap real yield (negative real yields = gold positive)
    real_yield_boost = 0.0
    if macro_snap is not None:
        try:
            # Try various attribute names from macro_snap dataclass
            real_yield = getattr(macro_snap, "real_yield_10y", None)
            if real_yield is None:
                real_yield = getattr(macro_snap, "real_yield", None)
            if real_yield is not None and isinstance(real_yield, (int, float)):
                import math
                if not math.isnan(real_yield) and not math.isinf(real_yield):
                    # Each -1% real yield adds ~10 pts
                    real_yield_boost = max(0.0, min(30.0, -real_yield * 10.0))
        except Exception:
            pass

    raw = (
        50.0                          # base
        + net_dove * 30.0             # dovish push
        + inflation_conf * 20.0       # inflation adds gold demand
        + real_yield_boost            # negative real yields support gold
    )
    return round(max(0.0, min(100.0, raw)), 2)


def _safe_haven_score(narratives: list, sentiment_snap, quant_composite: float) -> float:
    """
    Safe-haven demand score 0-100.
    Combines: safe-haven narrative + fear spike + low quant score (quant bearish = fear).
    """
    safe_conf = _get_narrative_confidence(narratives, "SAFE_HAVEN_BID")
    usd_conf = _get_narrative_confidence(narratives, "USD_DOMINANCE")

    fear_boost = 0.0
    if sentiment_snap is not None:
        fear_index = getattr(sentiment_snap, "fear_index", 0.0)
        fear_spike = getattr(sentiment_snap, "fear_spike", False)
        fear_boost = fear_index * 25.0 + (15.0 if fear_spike else 0.0)

    # Low quant score (< 40) suggests bearish quant → safe haven bid likely
    quant_boost = max(0.0, (50.0 - quant_composite) * 0.4)

    # USD dominance is a headwind for safe haven gold demand
    usd_drag = usd_conf * 15.0

    raw = (
        50.0
        + safe_conf * 30.0
        + fear_boost
        + quant_boost
        - usd_drag
    )
    return round(max(0.0, min(100.0, raw)), 2)


def _retail_score(narratives: list, sentiment_snap) -> float:
    """
    Retail/social conviction score 0-100.
    Central bank buying + inflation fear + positive sentiment = retail pile-in.
    Risk-on environment suppresses retail gold interest.
    """
    cb_conf = _get_narrative_confidence(narratives, "CENTRAL_BANK_BUYING")
    inflation_conf = _get_narrative_confidence(narratives, "INFLATION_FEAR")
    risk_on_conf = _get_narrative_confidence(narratives, "RISK_ON")

    optimism_boost = 0.0
    if sentiment_snap is not None:
        opt_index = getattr(sentiment_snap, "optimism_index", 0.0)
        # For gold: retail fear-driven optimism (not equity optimism) is additive
        polarity = getattr(sentiment_snap, "composite_polarity", 0.0)
        # Positive polarity = gold-positive sentiment
        optimism_boost = max(0.0, polarity * 20.0)

    raw = (
        40.0                          # lower base (retail is smaller component)
        + cb_conf * 30.0              # CB buying validates gold thesis
        + inflation_conf * 20.0       # inflation = hedge narrative
        + optimism_boost
        - risk_on_conf * 20.0         # risk-on pulls retail away from gold
    )
    return round(max(0.0, min(100.0, raw)), 2)


class GoldRiskIndex:
    """
    Computes the Gold Risk Index from narrative + sentiment + macro signals.
    Stateless — each call to compute() is independent.
    """

    def compute(
        self,
        narratives: list,
        sentiment_snap,
        macro_snap,
        recent_events: list,
        quant_composite: float = 50.0,
    ) -> GRISnapshot:
        """
        Compute full GRI snapshot.

        Args:
            narratives:      List of NarrativeSignal or dicts
            sentiment_snap:  SentimentEngineSnapshot or None
            macro_snap:      MacroSnapshot from quant engine or None
            recent_events:   List of (timestamp, message, color) tuples
            quant_composite: Current quant signal score 0-100 (default 50)

        Returns:
            GRISnapshot
        """
        try:
            geo = _geo_score(narratives, recent_events)
            monetary = _monetary_score(narratives, macro_snap)
            safe_haven = _safe_haven_score(narratives, sentiment_snap, quant_composite)
            retail = _retail_score(narratives, sentiment_snap)

            composite = (
                geo * _W_GEO
                + monetary * _W_MONETARY
                + safe_haven * _W_SAFE_HAVEN
                + retail * _W_RETAIL
            )
            composite = round(max(0.0, min(100.0, composite)), 2)

            return GRISnapshot(
                score=composite,
                geo_component=geo,
                monetary_component=monetary,
                safe_haven_component=safe_haven,
                retail_component=retail,
                label=_score_to_gri_label(composite),
            )

        except Exception as e:
            logger.warning("gri_compute_failed: %s", e)
            return GRISnapshot(
                score=50.0,
                geo_component=50.0,
                monetary_component=50.0,
                safe_haven_component=50.0,
                retail_component=50.0,
                label="moderate",
            )
