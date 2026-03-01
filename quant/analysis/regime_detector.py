"""Rule-based market regime detector."""

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


class Regime(Enum):
    NORMAL = "NORMAL"
    RISK_OFF = "RISK_OFF"
    HAWKISH_FED = "HAWKISH_FED"
    INFLATION_PANIC = "INFLATION_PANIC"
    WAR_ESCALATION = "WAR_ESCALATION"
    LIQUIDITY_SQUEEZE = "LIQUIDITY_SQUEEZE"


REGIME_COLORS = {
    Regime.NORMAL: "green",
    Regime.RISK_OFF: "yellow",
    Regime.HAWKISH_FED: "red",
    Regime.INFLATION_PANIC: "orange",
    Regime.WAR_ESCALATION: "bright_red",
    Regime.LIQUIDITY_SQUEEZE: "bright_red",
}

REGIME_GOLD_BIAS = {
    Regime.NORMAL: "neutral",
    Regime.RISK_OFF: "bullish",
    Regime.HAWKISH_FED: "bearish",
    Regime.INFLATION_PANIC: "bullish",
    Regime.WAR_ESCALATION: "bullish",
    Regime.LIQUIDITY_SQUEEZE: "mixed",
}

REGIME_DESCRIPTIONS = {
    Regime.NORMAL: "Markets are calm. Gold influenced by technical levels and DXY.",
    Regime.RISK_OFF: "Risk-off sentiment detected. Gold attracting safe-haven flows.",
    Regime.HAWKISH_FED: "Hawkish Fed/rising yields pressuring gold. Watch DXY closely.",
    Regime.INFLATION_PANIC: "Inflation fears rising. Gold as inflation hedge in play.",
    Regime.WAR_ESCALATION: "Geopolitical escalation. Safe-haven demand for gold elevated.",
    Regime.LIQUIDITY_SQUEEZE: "Liquidity crunch detected. Watch for forced gold selling.",
}


@dataclass
class RegimeSnapshot:
    regime: Regime
    gold_bias: str          # bullish / bearish / neutral / mixed
    description: str
    confidence: float       # 0.0 to 1.0
    triggers: list          # list of rule names that fired
    color: str              # for Rich terminal display
    timestamp: float = field(default_factory=time.time)


class RegimeDetector:
    """Detects market regime using priority-ordered rules."""

    def __init__(self):
        self.config = get_config()
        self._last_regime = Regime.NORMAL

    def detect(self, macro_data=None, geo_sentiment: float = 0.0,
               poly_risk_off_index: float = 0.0) -> RegimeSnapshot:
        """
        Detect regime. Priority order (highest → lowest):
        1. LIQUIDITY_SQUEEZE
        2. WAR_ESCALATION
        3. HAWKISH_FED
        4. RISK_OFF
        5. INFLATION_PANIC
        6. NORMAL
        """
        cfg = self.config
        triggers = []
        confidence = 0.5

        # Extract macro values safely
        vix = macro_data.vix.price if (macro_data and macro_data.vix) else None
        spy_5d = macro_data.spy.change_pct_5d if (macro_data and macro_data.spy) else None
        us10y = macro_data.us10y.price if (macro_data and macro_data.us10y) else None
        dxy_5d = macro_data.dxy.change_pct_5d if (macro_data and macro_data.dxy) else None
        uso_5d = macro_data.uso.change_pct_5d if (macro_data and macro_data.uso) else None

        # === Rule 1: LIQUIDITY_SQUEEZE (highest priority) ===
        lq_triggers = []
        if vix and vix > cfg.vix_liquidity_squeeze:
            lq_triggers.append(f"VIX={vix:.1f}>{cfg.vix_liquidity_squeeze}")
        if spy_5d and spy_5d < cfg.spy_liquidity_5d_change:
            lq_triggers.append(f"SPY={spy_5d:.1f}%<{cfg.spy_liquidity_5d_change}%")
        if len(lq_triggers) >= 2:
            return self._snap(Regime.LIQUIDITY_SQUEEZE, lq_triggers, min(0.5 + len(lq_triggers)*0.15, 0.95))

        # === Rule 2: WAR_ESCALATION ===
        war_triggers = []
        if geo_sentiment > cfg.geo_severity_war:
            war_triggers.append(f"geo_severity={geo_sentiment:.2f}>{cfg.geo_severity_war}")
        if poly_risk_off_index > cfg.polymarket_risk_off_index:
            war_triggers.append(f"poly_risk_off={poly_risk_off_index:.2f}>{cfg.polymarket_risk_off_index}")
        if len(war_triggers) >= 2:
            return self._snap(Regime.WAR_ESCALATION, war_triggers, min(0.5 + len(war_triggers)*0.2, 0.92))

        # === Rule 3: HAWKISH_FED ===
        hawkish_triggers = []
        if us10y and us10y > cfg.yield_hawkish:
            hawkish_triggers.append(f"US10Y={us10y:.2f}%>{cfg.yield_hawkish}%")
        if dxy_5d and dxy_5d > cfg.dxy_hawkish_5d_change:
            hawkish_triggers.append(f"DXY_5d={dxy_5d:.1f}%>{cfg.dxy_hawkish_5d_change}%")
        if len(hawkish_triggers) >= 2:
            return self._snap(Regime.HAWKISH_FED, hawkish_triggers, 0.80)
        elif len(hawkish_triggers) == 1:
            confidence = 0.55
            # Only fire if strong single trigger
            if us10y and us10y > 5.0:
                return self._snap(Regime.HAWKISH_FED, hawkish_triggers, 0.65)

        # === Rule 4: RISK_OFF ===
        ro_triggers = []
        if vix and vix > cfg.vix_risk_off:
            ro_triggers.append(f"VIX={vix:.1f}>{cfg.vix_risk_off}")
        if dxy_5d and dxy_5d > 0.5:
            ro_triggers.append(f"DXY_5d={dxy_5d:.1f}%>0.5% (safe haven USD)")
        if len(ro_triggers) >= 1:
            return self._snap(Regime.RISK_OFF, ro_triggers, min(0.5 + len(ro_triggers)*0.15, 0.85))

        # === Rule 5: INFLATION_PANIC ===
        inf_triggers = []
        if uso_5d and uso_5d > cfg.uso_inflation_5d_change:
            inf_triggers.append(f"USO_5d={uso_5d:.1f}%>{cfg.uso_inflation_5d_change}%")
        if us10y and us10y > 4.0:
            inf_triggers.append(f"US10Y={us10y:.2f}%>4.0%")
        if len(inf_triggers) >= 2:
            return self._snap(Regime.INFLATION_PANIC, inf_triggers, 0.72)

        # === Default: NORMAL ===
        return self._snap(Regime.NORMAL, ["No elevated regime signals detected"], 0.7)

    def _snap(self, regime: Regime, triggers: list, confidence: float) -> RegimeSnapshot:
        if regime != self._last_regime:
            logger.info("regime_changed", from_regime=self._last_regime.value, to_regime=regime.value, triggers=triggers)
        self._last_regime = regime
        return RegimeSnapshot(
            regime=regime,
            gold_bias=REGIME_GOLD_BIAS[regime],
            description=REGIME_DESCRIPTIONS[regime],
            confidence=confidence,
            triggers=triggers,
            color=REGIME_COLORS[regime],
        )
