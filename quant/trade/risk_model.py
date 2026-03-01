"""Volatility regime classification and position sizing."""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.logger import get_logger

logger = get_logger(__name__)


class VolatilityRegime(Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


VOL_REGIME_COLORS = {
    VolatilityRegime.LOW: "green",
    VolatilityRegime.NORMAL: "yellow",
    VolatilityRegime.HIGH: "orange1",
    VolatilityRegime.EXTREME: "bright_red",
}

# Position size factors per regime (fraction of full size)
VOL_POSITION_FACTORS = {
    VolatilityRegime.LOW: 1.0,
    VolatilityRegime.NORMAL: 0.75,
    VolatilityRegime.HIGH: 0.50,
    VolatilityRegime.EXTREME: 0.25,
}


@dataclass
class RiskSnapshot:
    atr14: float
    atr_pct_of_price: float         # ATR as % of current price
    atr_percentile: float           # 0-100, where ATR sits vs 30-day history
    vol_regime: VolatilityRegime
    vol_regime_color: str
    position_size_factor: float     # 0.25 to 1.0
    daily_range_estimate: float     # expected intraday range
    timestamp: float = field(default_factory=time.time)


class RiskModel:
    """Tracks ATR history and classifies volatility regime."""

    ATR_HISTORY_SIZE = 200  # ~30 days at 5-min bars per day

    def __init__(self):
        self._atr_history: list = []

    def compute(self, bars: list, current_price: float) -> Optional[RiskSnapshot]:
        """Compute risk snapshot from bar history."""
        if len(bars) < 15:
            return None

        try:
            import pandas as pd

            df = pd.DataFrame([
                {"high": b.high, "low": b.low, "close": b.close}
                for b in bars
            ])
            close = df["close"]
            prev_close = close.shift(1)
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr_series = tr.rolling(14).mean().dropna()
            if atr_series.empty:
                return None

            atr14 = float(atr_series.iloc[-1])
            atr_pct = (atr14 / current_price) * 100 if current_price > 0 else 0.0

            # Update ATR history
            self._atr_history.append(atr14)
            if len(self._atr_history) > self.ATR_HISTORY_SIZE:
                self._atr_history = self._atr_history[-self.ATR_HISTORY_SIZE:]

            # Compute ATR percentile
            if len(self._atr_history) >= 10:
                arr = np.array(self._atr_history)
                percentile = float(np.sum(arr <= atr14) / len(arr) * 100)
            else:
                percentile = 50.0

            # Classify regime
            if percentile < 25:
                regime = VolatilityRegime.LOW
            elif percentile < 75:
                regime = VolatilityRegime.NORMAL
            elif percentile < 95:
                regime = VolatilityRegime.HIGH
            else:
                regime = VolatilityRegime.EXTREME

            # Daily range estimate (intraday ATR × sqrt of bars per day)
            # Assume ~78 5-min bars per trading day (6.5h)
            bars_per_day = 78
            daily_range = atr14 * np.sqrt(bars_per_day / 14)

            return RiskSnapshot(
                atr14=round(atr14, 2),
                atr_pct_of_price=round(atr_pct, 4),
                atr_percentile=round(percentile, 1),
                vol_regime=regime,
                vol_regime_color=VOL_REGIME_COLORS[regime],
                position_size_factor=VOL_POSITION_FACTORS[regime],
                daily_range_estimate=round(daily_range, 2),
            )
        except Exception as e:
            logger.error("risk_model_error", error=str(e))
            return None
