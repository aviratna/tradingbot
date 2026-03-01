"""Macro environment scoring: DXY, yield, VIX, SPY → MacroSnapshot (0-100)."""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MacroSnapshot:
    # Inputs
    dxy_price: Optional[float] = None
    dxy_5d_change: Optional[float] = None
    us10y_price: Optional[float] = None
    spy_5d_change: Optional[float] = None
    vix_price: Optional[float] = None
    uso_5d_change: Optional[float] = None

    # Sub-scores (0-100 each)
    dxy_score: float = 50.0       # higher DXY = bearish for gold → lower score
    yield_score: float = 50.0     # higher real yields = bearish for gold → lower score
    vix_score: float = 50.0       # higher VIX = bullish for gold → higher score
    risk_score: float = 50.0      # SPY sell-off → gold safe haven → higher score

    # Composite macro score (0-100)
    score: float = 50.0

    # Human-readable assessment
    macro_bias: str = "neutral"   # bullish / bearish / neutral
    key_drivers: list = field(default_factory=list)

    timestamp: float = field(default_factory=time.time)


def compute_macro(macro_data) -> MacroSnapshot:
    """
    Score macro environment for gold.

    Gold is bullish when:
    - DXY is falling (inverse correlation)
    - Real yields are falling (or negative)
    - VIX is rising (fear → safe haven)
    - SPY is falling (risk-off)
    - Oil rising (inflation expectations)
    """
    snap = MacroSnapshot()
    drivers = []

    # --- DXY Score (inverted — higher DXY = worse for gold) ---
    if macro_data.dxy:
        dxy = macro_data.dxy.price
        dxy_5d = macro_data.dxy.change_pct_5d
        snap.dxy_price = dxy
        snap.dxy_5d_change = dxy_5d

        # Base: DXY level (100-105 neutral, >105 bearish, <98 bullish)
        if dxy < 98:
            dxy_score = 75.0
            drivers.append(f"Weak DXY ({dxy:.1f}) → gold bullish")
        elif dxy < 102:
            dxy_score = 55.0
        elif dxy < 106:
            dxy_score = 40.0
        else:
            dxy_score = 25.0
            drivers.append(f"Strong DXY ({dxy:.1f}) → gold headwind")

        # 5-day momentum adjustment
        if dxy_5d < -0.5:
            dxy_score += 10
            drivers.append(f"DXY falling ({dxy_5d:.1f}% 5d)")
        elif dxy_5d > 1.0:
            dxy_score -= 10
            drivers.append(f"DXY strengthening ({dxy_5d:.1f}% 5d)")

        snap.dxy_score = max(0, min(100, dxy_score))

    # --- Yield Score ---
    if macro_data.us10y:
        yield_rate = macro_data.us10y.price  # percentage (e.g., 4.3)
        snap.us10y_price = yield_rate

        # High real yields suppress gold
        if yield_rate < 3.5:
            yield_score = 70.0
            drivers.append(f"Low yields ({yield_rate:.2f}%) → gold friendly")
        elif yield_rate < 4.0:
            yield_score = 55.0
        elif yield_rate < 4.5:
            yield_score = 40.0
        else:
            yield_score = 25.0
            drivers.append(f"High yields ({yield_rate:.2f}%) → gold headwind")

        snap.yield_score = max(0, min(100, yield_score))

    # --- VIX Score (higher VIX = fear = gold bullish) ---
    if macro_data.vix:
        vix = macro_data.vix.price
        snap.vix_price = vix

        if vix > 35:
            vix_score = 90.0
            drivers.append(f"Extreme fear VIX={vix:.1f}")
        elif vix > 25:
            vix_score = 70.0
            drivers.append(f"High fear VIX={vix:.1f} → safe haven demand")
        elif vix > 18:
            vix_score = 55.0
        elif vix > 12:
            vix_score = 45.0
        else:
            vix_score = 35.0
            drivers.append(f"Low fear VIX={vix:.1f} → risk-on")

        snap.vix_score = max(0, min(100, vix_score))

    # --- SPY/Risk Score ---
    if macro_data.spy:
        spy_5d = macro_data.spy.change_pct_5d
        snap.spy_5d_change = spy_5d

        if spy_5d < -5:
            risk_score = 80.0
            drivers.append(f"Equity rout (SPY {spy_5d:.1f}% 5d) → flight to gold")
        elif spy_5d < -2:
            risk_score = 65.0
        elif spy_5d < 0:
            risk_score = 55.0
        elif spy_5d < 2:
            risk_score = 45.0
        else:
            risk_score = 35.0
            drivers.append(f"Risk-on (SPY +{spy_5d:.1f}% 5d) → gold less favored")

        # Oil inflation signal
        if macro_data.uso:
            uso_5d = macro_data.uso.change_pct_5d
            snap.uso_5d_change = uso_5d
            if uso_5d > 3:
                risk_score = min(risk_score + 10, 100)
                drivers.append(f"Oil surge ({uso_5d:.1f}% 5d) → inflation fear")

        snap.risk_score = max(0, min(100, risk_score))

    # --- Composite Macro Score ---
    # Weighted average of sub-scores
    weights = {"dxy": 0.30, "yield": 0.25, "vix": 0.25, "risk": 0.20}
    composite = (
        snap.dxy_score * weights["dxy"] +
        snap.yield_score * weights["yield"] +
        snap.vix_score * weights["vix"] +
        snap.risk_score * weights["risk"]
    )
    snap.score = round(composite, 2)

    # Overall bias
    if snap.score >= 60:
        snap.macro_bias = "bullish"
    elif snap.score <= 40:
        snap.macro_bias = "bearish"
    else:
        snap.macro_bias = "neutral"

    snap.key_drivers = drivers[:4]  # top 4
    return snap
