"""Trade adapter: risk control multipliers from OSINT signals.

IMPORTANT: This module provides signal multipliers ONLY.
It does NOT execute trades, place orders, or connect to any broker.
All output is advisory/informational for the quant dashboard.

Risk rules:
  BOOST:  GRI > 85 AND fear_spike AND quant_bullish    → size_multiplier = 1.2
  REDUCE: hawkish_index > 75 AND USD_DOMINANCE dominant → size_multiplier = 0.6
  CAUTION: GRI > 70 (elevated) AND quant_composite < 45  → size_multiplier = 0.8
  DEFAULT: no significant signal                         → size_multiplier = 1.0
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeBias:
    rule_name: str            # identifier for which rule triggered
    size_multiplier: float    # 0.0..2.0 (1.0 = no change)
    direction_bias: str       # "LONG_FAVORED" | "SHORT_FAVORED" | "NEUTRAL"
    rationale: str            # human-readable explanation
    confidence: float         # 0..1 — how strongly the rule is triggered
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Rule helpers
# ---------------------------------------------------------------------------

def _is_quant_bullish(quant_direction: str, quant_composite: float) -> bool:
    """True if quant engine is pointing bullish."""
    if quant_direction and quant_direction.upper() in ("LONG", "BULLISH", "BUY"):
        return True
    return quant_composite >= 55.0


def _get_narrative_confidence(narratives: list, name: str) -> float:
    """Extract confidence for a named narrative."""
    for n in narratives:
        if isinstance(n, dict):
            if n.get("name") == name:
                return float(n.get("confidence", 0.0))
        else:
            if getattr(n, "name", None) == name:
                return float(getattr(n, "confidence", 0.0))
    return 0.0


def _dominant_narrative_name(narratives: list) -> Optional[str]:
    """Return name of the highest-confidence narrative, or None."""
    if not narratives:
        return None
    best = None
    best_conf = 0.0
    for n in narratives:
        if isinstance(n, dict):
            conf = float(n.get("confidence", 0.0))
            name = n.get("name", "")
        else:
            conf = float(getattr(n, "confidence", 0.0))
            name = getattr(n, "name", "")
        if conf > best_conf:
            best_conf = conf
            best = name
    return best


class TradeAdapter:
    """
    Evaluates OSINT + quant signals against risk control rules.
    Returns a list of TradeBias objects (typically 1, the highest-priority rule).

    This is purely advisory — no broker connections, no order management.
    """

    # Rule priority: first match wins (highest priority first)
    _RULES = [
        "LONG_MULTIPLIER_BOOST",
        "HAWKISH_USD_REDUCE",
        "ELEVATED_GRI_CAUTION",
        "DEFAULT",
    ]

    def compute_bias(
        self,
        gri_snap,
        fast_snap,
        narratives: list,
        hawkish_index: float,
        quant_direction: str = "NEUTRAL",
        quant_composite: float = 50.0,
    ) -> List[TradeBias]:
        """
        Evaluate risk control rules and return advisory biases.

        Args:
            gri_snap:         GRISnapshot
            fast_snap:        FastSignalSnapshot
            narratives:       List of NarrativeSignal or dicts
            hawkish_index:    0-100 (100 = fully hawkish)
            quant_direction:  "LONG" | "SHORT" | "NEUTRAL"
            quant_composite:  0-100 quant signal score

        Returns:
            List[TradeBias] — usually 1 entry (the active rule)
        """
        try:
            gri_score = getattr(gri_snap, "score", 50.0) if gri_snap else 50.0
            fear_spike = False
            if fast_snap is not None:
                # fear_spike lives in sentiment, not fast_snap, but OsintAggregator
                # passes it through as attribute if it wraps it
                fear_spike = getattr(fast_snap, "fear_spike", False)

            # --- Rule 1: LONG_MULTIPLIER_BOOST ---
            if (
                gri_score > 85.0
                and fear_spike
                and _is_quant_bullish(quant_direction, quant_composite)
            ):
                conf = min(1.0, (gri_score - 85.0) / 15.0 * 0.5 + 0.5)
                return [TradeBias(
                    rule_name="LONG_MULTIPLIER_BOOST",
                    size_multiplier=1.2,
                    direction_bias="LONG_FAVORED",
                    rationale=(
                        f"Extreme GRI ({gri_score:.0f}) + fear spike + quant bullish "
                        "→ institutional-grade long setup"
                    ),
                    confidence=round(conf, 3),
                )]

            # --- Rule 2: HAWKISH_USD_REDUCE ---
            usd_conf = _get_narrative_confidence(narratives, "USD_DOMINANCE")
            dominant = _dominant_narrative_name(narratives)
            if hawkish_index > 75.0 and dominant == "USD_DOMINANCE":
                conf = min(1.0, (hawkish_index - 75.0) / 25.0 * 0.7 + 0.3)
                return [TradeBias(
                    rule_name="HAWKISH_USD_REDUCE",
                    size_multiplier=0.6,
                    direction_bias="SHORT_FAVORED",
                    rationale=(
                        f"Hawkish index {hawkish_index:.0f}/100 + USD dominance narrative "
                        "→ reduce long exposure"
                    ),
                    confidence=round(conf, 3),
                )]

            # --- Rule 3: ELEVATED_GRI_CAUTION ---
            if gri_score > 70.0 and quant_composite < 45.0:
                conf = min(1.0, (gri_score - 70.0) / 30.0 * 0.5 + 0.3)
                return [TradeBias(
                    rule_name="ELEVATED_GRI_CAUTION",
                    size_multiplier=0.8,
                    direction_bias="NEUTRAL",
                    rationale=(
                        f"Elevated GRI ({gri_score:.0f}) with weak quant ({quant_composite:.0f}) "
                        "→ cautious sizing"
                    ),
                    confidence=round(conf, 3),
                )]

            # --- Rule 4: DEFAULT ---
            return [TradeBias(
                rule_name="DEFAULT",
                size_multiplier=1.0,
                direction_bias="NEUTRAL",
                rationale="No high-conviction OSINT risk control trigger active",
                confidence=0.5,
            )]

        except Exception as e:
            logger.warning("trade_adapter_compute_failed: %s", e)
            return [TradeBias(
                rule_name="ERROR",
                size_multiplier=1.0,
                direction_bias="NEUTRAL",
                rationale=f"Adapter error: {e}",
                confidence=0.0,
            )]
