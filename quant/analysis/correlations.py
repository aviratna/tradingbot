"""Rolling correlation analysis: XAU vs macro assets, regime shift detection."""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque
from collections import deque
import numpy as np
import pandas as pd
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.logger import get_logger

logger = get_logger(__name__)

# Assets to correlate with XAU
CORR_ASSETS = ["DXY", "SPY", "VIX", "USO", "XAUT"]


@dataclass
class CorrelationSnapshot:
    # Correlation coefficients (-1 to 1) XAU vs each asset
    xau_dxy: Optional[float] = None
    xau_spy: Optional[float] = None
    xau_vix: Optional[float] = None
    xau_uso: Optional[float] = None
    xau_xaut: Optional[float] = None

    # Regime shift flags
    regime_shift_detected: bool = False
    shift_description: str = ""

    # Previous correlations for delta calculation
    prev_xau_dxy: Optional[float] = None
    prev_xau_vix: Optional[float] = None

    data_points: int = 0
    timestamp: float = field(default_factory=time.time)


class CorrelationAnalyzer:
    """Maintains rolling price history and computes correlations."""

    WINDOW = 30  # days of daily closes for correlation

    def __init__(self):
        # Rolling daily close prices (deque of float)
        self._xau_prices: Deque[float] = deque(maxlen=self.WINDOW)
        self._dxy_prices: Deque[float] = deque(maxlen=self.WINDOW)
        self._spy_prices: Deque[float] = deque(maxlen=self.WINDOW)
        self._vix_prices: Deque[float] = deque(maxlen=self.WINDOW)
        self._uso_prices: Deque[float] = deque(maxlen=self.WINDOW)
        self._xaut_prices: Deque[float] = deque(maxlen=self.WINDOW)

        self._prev_snapshot: Optional[CorrelationSnapshot] = None
        self._last_update: float = 0

    def update_xau(self, price: float) -> None:
        """Add XAU price point (call on daily close)."""
        self._xau_prices.append(price)
        self._last_update = time.time()

    def update_macro(self, dxy: Optional[float], spy: Optional[float],
                     vix: Optional[float], uso: Optional[float]) -> None:
        """Update macro asset prices."""
        if dxy:
            self._dxy_prices.append(dxy)
        if spy:
            self._spy_prices.append(spy)
        if vix:
            self._vix_prices.append(vix)
        if uso:
            self._uso_prices.append(uso)

    def update_xaut(self, price: float) -> None:
        """Update XAUT price."""
        self._xaut_prices.append(price)

    @staticmethod
    def _corr(a: Deque[float], b: Deque[float]) -> Optional[float]:
        """Compute Pearson correlation coefficient between two deques."""
        min_len = min(len(a), len(b))
        if min_len < 5:
            return None
        arr_a = np.array(list(a)[-min_len:])
        arr_b = np.array(list(b)[-min_len:])
        if arr_a.std() == 0 or arr_b.std() == 0:
            return None
        return float(np.corrcoef(arr_a, arr_b)[0, 1])

    def _detect_regime_shift(self, snap: CorrelationSnapshot) -> tuple:
        """Detect if a correlation regime shift has occurred."""
        if not self._prev_snapshot:
            return False, ""

        shift_threshold = 0.4
        descriptions = []

        if snap.xau_dxy is not None and self._prev_snapshot.xau_dxy is not None:
            delta = abs(snap.xau_dxy - self._prev_snapshot.xau_dxy)
            if delta > shift_threshold:
                direction = "strengthened" if snap.xau_dxy > self._prev_snapshot.xau_dxy else "weakened"
                descriptions.append(f"XAU-DXY correlation {direction} by {delta:.2f}")

        if snap.xau_vix is not None and self._prev_snapshot.xau_vix is not None:
            delta = abs(snap.xau_vix - self._prev_snapshot.xau_vix)
            if delta > shift_threshold:
                direction = "strengthened" if snap.xau_vix > self._prev_snapshot.xau_vix else "weakened"
                descriptions.append(f"XAU-VIX correlation {direction} by {delta:.2f}")

        if descriptions:
            return True, "; ".join(descriptions)
        return False, ""

    def compute(self) -> CorrelationSnapshot:
        """Compute current correlation snapshot."""
        snap = CorrelationSnapshot(
            xau_dxy=self._corr(self._xau_prices, self._dxy_prices),
            xau_spy=self._corr(self._xau_prices, self._spy_prices),
            xau_vix=self._corr(self._xau_prices, self._vix_prices),
            xau_uso=self._corr(self._xau_prices, self._uso_prices),
            xau_xaut=self._corr(self._xau_prices, self._xaut_prices),
            data_points=len(self._xau_prices),
        )

        shift_detected, shift_desc = self._detect_regime_shift(snap)
        snap.regime_shift_detected = shift_detected
        snap.shift_description = shift_desc

        if shift_detected:
            logger.info("correlation_regime_shift", description=shift_desc)

        self._prev_snapshot = snap
        return snap
