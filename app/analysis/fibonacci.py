"""Fibonacci analysis for identifying support/resistance and patterns."""
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from ..config import settings


class TrendDirection(Enum):
    """Trend direction enumeration."""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"


@dataclass
class FibonacciLevel:
    """Fibonacci level with price and significance."""
    ratio: float
    price: float
    label: str
    is_support: bool
    is_resistance: bool
    strength: float  # 0-1 indicating historical significance


@dataclass
class FibonacciAnalysis:
    """Complete Fibonacci analysis result."""
    symbol: str
    trend: TrendDirection
    swing_high: float
    swing_low: float
    current_price: float
    retracement_levels: List[FibonacciLevel]
    extension_levels: List[FibonacciLevel]
    current_zone: str
    nearest_support: Optional[FibonacciLevel]
    nearest_resistance: Optional[FibonacciLevel]
    pattern_detected: Optional[str]
    confidence: float


class FibonacciAnalyzer:
    """
    Fibonacci analysis for identifying support/resistance levels and patterns.

    Uses the golden ratio (1.618) and related Fibonacci ratios to identify
    potential reversal points and price targets.
    """

    # Standard Fibonacci retracement ratios
    RETRACEMENT_RATIOS = {
        0.0: "0% (Swing High/Low)",
        0.236: "23.6%",
        0.382: "38.2%",
        0.5: "50%",
        0.618: "61.8% (Golden Ratio)",
        0.786: "78.6%",
        1.0: "100% (Swing Low/High)"
    }

    # Fibonacci extension ratios
    EXTENSION_RATIOS = {
        1.0: "100%",
        1.272: "127.2%",
        1.414: "141.4%",
        1.618: "161.8% (Golden Extension)",
        2.0: "200%",
        2.618: "261.8%",
        3.618: "361.8%"
    }

    def __init__(self, lookback_period: int = 50):
        """
        Initialize Fibonacci analyzer.

        Args:
            lookback_period: Number of periods to look back for swing points
        """
        self.lookback_period = lookback_period

    def find_swing_points(
        self,
        prices: pd.Series,
        window: int = 5
    ) -> Tuple[float, float, int, int]:
        """
        Find swing high and swing low points in price data.

        Args:
            prices: Series of price data
            window: Window size for local max/min detection

        Returns:
            Tuple of (swing_high, swing_low, high_index, low_index)
        """
        if len(prices) < window * 2:
            return prices.max(), prices.min(), prices.idxmax(), prices.idxmin()

        # Find local maxima and minima
        rolling_max = prices.rolling(window=window, center=True).max()
        rolling_min = prices.rolling(window=window, center=True).min()

        # Identify swing points
        swing_highs = prices[prices == rolling_max]
        swing_lows = prices[prices == rolling_min]

        if len(swing_highs) == 0 or len(swing_lows) == 0:
            return prices.max(), prices.min(), prices.idxmax(), prices.idxmin()

        # Get the most significant recent swing points
        swing_high = swing_highs.max()
        swing_low = swing_lows.min()
        high_idx = swing_highs.idxmax()
        low_idx = swing_lows.idxmin()

        return swing_high, swing_low, high_idx, low_idx

    def detect_trend(
        self,
        prices: pd.Series,
        swing_high_idx,
        swing_low_idx
    ) -> TrendDirection:
        """
        Detect trend direction based on swing points.

        Args:
            prices: Series of price data
            swing_high_idx: Index of swing high
            swing_low_idx: Index of swing low

        Returns:
            TrendDirection enum
        """
        if len(prices) < 2:
            return TrendDirection.SIDEWAYS

        # Compare positions of swing high and low
        # If high came after low, we're in uptrend (price went up)
        # If low came after high, we're in downtrend (price went down)
        try:
            high_pos = list(prices.index).index(swing_high_idx)
            low_pos = list(prices.index).index(swing_low_idx)

            if high_pos > low_pos:
                return TrendDirection.UPTREND
            elif low_pos > high_pos:
                return TrendDirection.DOWNTREND
        except (ValueError, IndexError):
            pass

        # Fallback: check if current price is closer to high or low
        current = prices.iloc[-1]
        swing_high = prices.max()
        swing_low = prices.min()
        mid = (swing_high + swing_low) / 2

        if current > mid:
            return TrendDirection.UPTREND
        elif current < mid:
            return TrendDirection.DOWNTREND

        return TrendDirection.SIDEWAYS

    def calculate_retracement_levels(
        self,
        swing_high: float,
        swing_low: float,
        trend: TrendDirection
    ) -> List[FibonacciLevel]:
        """
        Calculate Fibonacci retracement levels.

        Args:
            swing_high: Swing high price
            swing_low: Swing low price
            trend: Current trend direction

        Returns:
            List of FibonacciLevel objects
        """
        levels = []
        price_range = swing_high - swing_low

        for ratio, label in self.RETRACEMENT_RATIOS.items():
            if trend == TrendDirection.UPTREND:
                # In uptrend, retracements are measured from high going down
                price = swing_high - (price_range * ratio)
                is_support = ratio > 0
                is_resistance = ratio == 0
            else:
                # In downtrend, retracements are measured from low going up
                price = swing_low + (price_range * ratio)
                is_support = ratio == 0
                is_resistance = ratio > 0

            # Key levels have higher strength
            strength = 1.0 if ratio in [0.382, 0.5, 0.618] else 0.7

            levels.append(FibonacciLevel(
                ratio=ratio,
                price=round(price, 4),
                label=label,
                is_support=is_support,
                is_resistance=is_resistance,
                strength=strength
            ))

        return sorted(levels, key=lambda x: x.price, reverse=True)

    def calculate_extension_levels(
        self,
        swing_high: float,
        swing_low: float,
        trend: TrendDirection
    ) -> List[FibonacciLevel]:
        """
        Calculate Fibonacci extension levels.

        Args:
            swing_high: Swing high price
            swing_low: Swing low price
            trend: Current trend direction

        Returns:
            List of FibonacciLevel objects
        """
        levels = []
        price_range = swing_high - swing_low

        for ratio, label in self.EXTENSION_RATIOS.items():
            if trend == TrendDirection.UPTREND:
                # Extensions project above the swing high
                price = swing_low + (price_range * ratio)
                is_resistance = True
                is_support = False
            else:
                # Extensions project below the swing low
                price = swing_high - (price_range * ratio)
                is_resistance = False
                is_support = True

            # Key extension levels
            strength = 1.0 if ratio in [1.618, 2.618] else 0.6

            levels.append(FibonacciLevel(
                ratio=ratio,
                price=round(price, 4),
                label=label,
                is_support=is_support,
                is_resistance=is_resistance,
                strength=strength
            ))

        return sorted(levels, key=lambda x: x.price, reverse=True)

    def find_current_zone(
        self,
        current_price: float,
        levels: List[FibonacciLevel]
    ) -> str:
        """
        Find which Fibonacci zone the current price is in.

        Args:
            current_price: Current market price
            levels: List of Fibonacci levels

        Returns:
            Description of current zone
        """
        sorted_levels = sorted(levels, key=lambda x: x.price)

        for i in range(len(sorted_levels) - 1):
            lower = sorted_levels[i]
            upper = sorted_levels[i + 1]

            if lower.price <= current_price <= upper.price:
                return f"Between {lower.label} ({lower.price}) and {upper.label} ({upper.price})"

        if current_price < sorted_levels[0].price:
            return f"Below {sorted_levels[0].label}"
        else:
            return f"Above {sorted_levels[-1].label}"

    def find_nearest_levels(
        self,
        current_price: float,
        levels: List[FibonacciLevel]
    ) -> Tuple[Optional[FibonacciLevel], Optional[FibonacciLevel]]:
        """
        Find nearest support and resistance levels.

        Args:
            current_price: Current market price
            levels: List of Fibonacci levels

        Returns:
            Tuple of (nearest_support, nearest_resistance)
        """
        supports = [l for l in levels if l.price < current_price and l.is_support]
        resistances = [l for l in levels if l.price > current_price and l.is_resistance]

        nearest_support = max(supports, key=lambda x: x.price) if supports else None
        nearest_resistance = min(resistances, key=lambda x: x.price) if resistances else None

        return nearest_support, nearest_resistance

    def detect_pattern(
        self,
        prices: pd.Series,
        retracement_levels: List[FibonacciLevel]
    ) -> Optional[str]:
        """
        Detect common Fibonacci-based patterns.

        Args:
            prices: Price series
            retracement_levels: Calculated retracement levels

        Returns:
            Pattern name if detected, None otherwise
        """
        if len(prices) < 10:
            return None

        current = prices.iloc[-1]
        recent_prices = prices.tail(10)

        # Get key levels
        level_382 = next((l for l in retracement_levels if l.ratio == 0.382), None)
        level_618 = next((l for l in retracement_levels if l.ratio == 0.618), None)
        level_50 = next((l for l in retracement_levels if l.ratio == 0.5), None)

        if not all([level_382, level_618, level_50]):
            return None

        # Check for bounce off 61.8% level (golden ratio)
        tolerance = 0.02 * current  # 2% tolerance
        if abs(recent_prices.min() - level_618.price) < tolerance:
            if current > level_618.price:
                return "Golden Ratio Bounce (61.8% support)"

        # Check for rejection at 38.2% level
        if abs(recent_prices.max() - level_382.price) < tolerance:
            if current < level_382.price:
                return "38.2% Rejection (potential continuation)"

        # Check for consolidation at 50% level
        if abs(current - level_50.price) < tolerance:
            if recent_prices.std() < 0.01 * current:
                return "50% Consolidation Zone"

        return None

    def analyze(
        self,
        prices: pd.Series,
        symbol: str = "UNKNOWN"
    ) -> FibonacciAnalysis:
        """
        Perform complete Fibonacci analysis on price data.

        Args:
            prices: Series of price data (OHLC close prices)
            symbol: Asset symbol for labeling

        Returns:
            FibonacciAnalysis object with complete analysis
        """
        if len(prices) < 5:
            raise ValueError("Insufficient price data for Fibonacci analysis")

        # Find swing points
        swing_high, swing_low, high_idx, low_idx = self.find_swing_points(prices)

        # Detect trend
        trend = self.detect_trend(prices, high_idx, low_idx)

        # Calculate levels
        retracement_levels = self.calculate_retracement_levels(
            swing_high, swing_low, trend
        )
        extension_levels = self.calculate_extension_levels(
            swing_high, swing_low, trend
        )

        current_price = prices.iloc[-1]
        all_levels = retracement_levels + extension_levels

        # Find current zone and nearest levels
        current_zone = self.find_current_zone(current_price, retracement_levels)
        nearest_support, nearest_resistance = self.find_nearest_levels(
            current_price, all_levels
        )

        # Detect patterns
        pattern = self.detect_pattern(prices, retracement_levels)

        # Calculate confidence based on data quality
        confidence = min(1.0, len(prices) / 100)  # More data = higher confidence

        return FibonacciAnalysis(
            symbol=symbol,
            trend=trend,
            swing_high=round(swing_high, 4),
            swing_low=round(swing_low, 4),
            current_price=round(current_price, 4),
            retracement_levels=retracement_levels,
            extension_levels=extension_levels,
            current_zone=current_zone,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            pattern_detected=pattern,
            confidence=confidence
        )

    def get_levels_as_dict(
        self,
        analysis: FibonacciAnalysis
    ) -> Dict[str, List[Dict]]:
        """
        Convert Fibonacci analysis to dictionary format for JSON serialization.

        Args:
            analysis: FibonacciAnalysis object

        Returns:
            Dictionary with retracement and extension levels
        """
        return {
            "symbol": analysis.symbol,
            "trend": analysis.trend.value,
            "swing_high": analysis.swing_high,
            "swing_low": analysis.swing_low,
            "current_price": analysis.current_price,
            "current_zone": analysis.current_zone,
            "pattern": analysis.pattern_detected,
            "confidence": analysis.confidence,
            "retracement_levels": [
                {
                    "ratio": l.ratio,
                    "price": l.price,
                    "label": l.label,
                    "is_support": l.is_support,
                    "is_resistance": l.is_resistance,
                    "strength": l.strength
                }
                for l in analysis.retracement_levels
            ],
            "extension_levels": [
                {
                    "ratio": l.ratio,
                    "price": l.price,
                    "label": l.label,
                    "strength": l.strength
                }
                for l in analysis.extension_levels
            ],
            "nearest_support": {
                "price": analysis.nearest_support.price,
                "label": analysis.nearest_support.label
            } if analysis.nearest_support else None,
            "nearest_resistance": {
                "price": analysis.nearest_resistance.price,
                "label": analysis.nearest_resistance.label
            } if analysis.nearest_resistance else None
        }
