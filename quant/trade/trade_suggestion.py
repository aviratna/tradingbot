"""ATR-based trade suggestion generator."""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeSuggestion:
    # Direction
    action: str                     # BUY / SELL / HOLD

    # Price levels
    current_price: float
    entry_low: float                # entry zone low
    entry_high: float               # entry zone high
    entry_mid: float                # midpoint of entry zone
    stop_loss: float
    take_profit_1: float
    take_profit_2: float

    # Risk metrics
    r_r_ratio: float                # reward/risk ratio to TP1
    atr14: float
    position_size_factor: float     # 0.25–1.0 based on volatility regime

    # Context
    regime: str
    signal_score: float             # composite 0-100
    confidence: float               # 0.0-1.0
    rationale: List[str]            # human-readable reasons

    # Timestamps
    timestamp: float = field(default_factory=time.time)
    valid_for_minutes: int = 30     # suggestion validity window


def generate_suggestion(
    current_price: float,
    tech_snap=None,
    signal_snap=None,
    risk_snap=None,
    regime_snap=None,
    forecast_snap=None,
) -> Optional[TradeSuggestion]:
    """Generate a structured trade suggestion based on current signals."""
    if not tech_snap or not signal_snap:
        return None

    try:
        atr14 = tech_snap.atr14 if tech_snap.atr14 > 0 else current_price * 0.005
        direction = signal_snap.direction.value
        composite = signal_snap.composite
        position_factor = risk_snap.position_size_factor if risk_snap else 0.75
        regime_name = regime_snap.regime.value if regime_snap else "NORMAL"

        rationale = []

        # Determine action
        if composite >= 55:
            action = "BUY"
        elif composite <= 45:
            action = "SELL"
        else:
            action = "HOLD"

        # Entry zone: ±0.3 × ATR around current price
        from quant.core.config import get_config
        cfg = get_config()
        entry_low = current_price - cfg.entry_atr_mult * atr14
        entry_high = current_price + cfg.entry_atr_mult * atr14
        entry_mid = (entry_low + entry_high) / 2

        # Stop loss and targets based on action
        if action == "BUY":
            stop_loss = entry_low - cfg.sl_atr_mult * atr14
            tp1 = entry_high + cfg.tp1_atr_mult * atr14
            tp2 = entry_high + cfg.tp2_atr_mult * atr14
        elif action == "SELL":
            stop_loss = entry_high + cfg.sl_atr_mult * atr14
            tp1 = entry_low - cfg.tp1_atr_mult * atr14
            tp2 = entry_low - cfg.tp2_atr_mult * atr14
        else:
            stop_loss = current_price - cfg.sl_atr_mult * atr14
            tp1 = current_price + cfg.tp1_atr_mult * atr14
            tp2 = current_price + cfg.tp2_atr_mult * atr14

        # R:R ratio
        sl_distance = abs(entry_mid - stop_loss)
        tp1_distance = abs(tp1 - entry_mid)
        rr_ratio = tp1_distance / sl_distance if sl_distance > 0 else 0.0

        # Build rationale
        if tech_snap:
            rationale.append(f"Trend: {tech_snap.trend} | RSI: {tech_snap.rsi14:.1f} ({tech_snap.rsi_signal})")
            rationale.append(f"MACD: {tech_snap.macd_signal_dir}")
        if signal_snap:
            rationale.append(
                f"Composite score: {composite:.1f} "
                f"[T:{signal_snap.tech_score:.0f} M:{signal_snap.macro_score:.0f} "
                f"S:{signal_snap.sentiment_score:.0f} P:{signal_snap.polymarket_score:.0f}]"
            )
        if regime_snap:
            rationale.append(f"Regime: {regime_name} → gold {regime_snap.gold_bias}")
        if forecast_snap:
            fc = forecast_snap.scenarios[1] if len(forecast_snap.scenarios) > 1 else None
            if fc:
                rationale.append(f"30-min forecast: ${fc.base_forecast:,.2f} [{fc.lower_band:,.2f}–{fc.upper_band:,.2f}]")
        if risk_snap:
            rationale.append(f"Vol regime: {risk_snap.vol_regime.value} (ATR%ile: {risk_snap.atr_percentile:.0f}th) → {position_factor:.0%} size")

        return TradeSuggestion(
            action=action,
            current_price=current_price,
            entry_low=round(entry_low, 2),
            entry_high=round(entry_high, 2),
            entry_mid=round(entry_mid, 2),
            stop_loss=round(stop_loss, 2),
            take_profit_1=round(tp1, 2),
            take_profit_2=round(tp2, 2),
            r_r_ratio=round(rr_ratio, 2),
            atr14=round(atr14, 2),
            position_size_factor=position_factor,
            regime=regime_name,
            signal_score=composite,
            confidence=signal_snap.confidence,
            rationale=rationale,
        )

    except Exception as e:
        logger.error("trade_suggestion_error", error=str(e))
        return None
