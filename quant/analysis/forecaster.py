"""Short-term price forecasting using ARIMA or rolling statistics."""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ForecastScenario:
    horizon_minutes: int
    base_forecast: float
    upper_band: float   # 80% confidence
    lower_band: float   # 80% confidence
    method: str         # "arima" or "rolling_stats"


@dataclass
class ForecastSnapshot:
    current_price: float
    scenarios: List[ForecastScenario]      # 15, 30, 60 min
    volatility_pct: float                   # ATR as % of price
    forecast_direction: str                 # up / down / flat
    confidence: float                       # 0.0 to 1.0
    method_used: str
    timestamp: float = field(default_factory=time.time)


def compute_forecast(bars: list, current_price: float) -> Optional[ForecastSnapshot]:
    """Generate price forecast using ARIMA or rolling statistics fallback."""
    if len(bars) < 30:
        return None

    closes = np.array([b.close for b in bars[-100:]])
    method_used = "rolling_stats"

    try:
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            import pandas as pd
            closes_series = pd.Series(closes)
            model = ARIMA(closes_series, order=(1, 1, 1))
            result = model.fit()
            forecast_30 = result.forecast(steps=6)  # 6 × 5min = 30 min
            arima_price_30 = float(forecast_30.iloc[-1])
            method_used = "arima"
    except Exception as e:
        logger.debug("arima_failed_using_rolling", error=str(e))
        arima_price_30 = None

    # Rolling statistics fallback / supplement
    returns = np.diff(closes) / closes[:-1]
    mean_return = float(np.mean(returns[-20:]))  # recent 20 bars
    std_return = float(np.std(returns[-20:]))
    vol_pct = std_return * 100

    def _rolling_forecast(steps: int) -> tuple:
        """Project price forward using mean reversion + momentum."""
        projected = current_price * (1 + mean_return) ** steps
        uncertainty = current_price * std_return * np.sqrt(steps)
        return projected, projected + 1.28 * uncertainty, projected - 1.28 * uncertainty

    scenarios = []

    for horizon_min, steps in [(15, 3), (30, 6), (60, 12)]:
        base, upper, lower = _rolling_forecast(steps)

        # Override 30-min base with ARIMA if available
        if arima_price_30 and horizon_min == 30:
            base = arima_price_30
            # Recalculate bands around ARIMA forecast
            uncertainty = current_price * std_return * np.sqrt(steps)
            upper = base + 1.28 * uncertainty
            lower = base - 1.28 * uncertainty

        scenarios.append(ForecastScenario(
            horizon_minutes=horizon_min,
            base_forecast=round(base, 2),
            upper_band=round(upper, 2),
            lower_band=round(lower, 2),
            method=method_used if horizon_min == 30 else "rolling_stats",
        ))

    # Direction based on 30-min forecast
    fc_30 = scenarios[1].base_forecast if len(scenarios) > 1 else current_price
    diff_pct = (fc_30 - current_price) / current_price * 100 if current_price > 0 else 0

    if diff_pct > 0.05:
        direction = "up"
    elif diff_pct < -0.05:
        direction = "down"
    else:
        direction = "flat"

    # Confidence based on data quantity and ARIMA success
    base_confidence = min(len(bars) / 100, 1.0) * 0.7
    if method_used == "arima":
        base_confidence = min(base_confidence + 0.2, 0.85)

    return ForecastSnapshot(
        current_price=current_price,
        scenarios=scenarios,
        volatility_pct=round(vol_pct, 4),
        forecast_direction=direction,
        confidence=round(base_confidence, 3),
        method_used=method_used,
    )
