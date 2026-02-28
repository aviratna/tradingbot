"""
Bank/Institutional Manipulation Detection for Precious Metals Trading.
Detects stop hunts, liquidity sweeps, and other manipulation patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ManipulationType(Enum):
    """Types of market manipulation patterns."""
    STOP_HUNT = "stop_hunt"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    VOLUME_ANOMALY = "volume_anomaly"
    SESSION_MANIPULATION = "session_manipulation"
    GAP_FILL = "gap_fill"
    ORDER_BLOCK = "order_block"
    FAKE_BREAKOUT = "fake_breakout"
    WHIPSAW = "whipsaw"


class ManipulationSeverity(Enum):
    """Severity levels for detected manipulation."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class KeyLevel:
    """Key price level for manipulation detection."""
    price: float
    level_type: str  # SUPPORT, RESISTANCE, SWING_HIGH, SWING_LOW, ROUND_NUMBER
    strength: float  # 0-1, how strong the level is
    touches: int  # Number of times price has touched this level
    last_touch: Optional[datetime] = None


@dataclass
class ManipulationAlert:
    """Alert for detected manipulation."""
    manipulation_type: ManipulationType
    severity: ManipulationSeverity
    timestamp: datetime
    price_at_detection: float
    description: str
    key_level_involved: Optional[float] = None
    expected_reversal: Optional[str] = None  # UP, DOWN
    confidence: float = 0.0
    expires_in_minutes: int = 15


@dataclass
class OrderBlock:
    """Institutional order block detection."""
    price_high: float
    price_low: float
    block_type: str  # BULLISH, BEARISH
    strength: float
    timestamp: datetime
    is_tested: bool = False
    is_broken: bool = False


@dataclass
class ManipulationAnalysis:
    """Complete manipulation analysis result."""
    symbol: str
    timestamp: datetime
    current_price: float
    active_alerts: list[ManipulationAlert] = field(default_factory=list)
    key_levels: list[KeyLevel] = field(default_factory=list)
    order_blocks: list[OrderBlock] = field(default_factory=list)
    session_risk: str = "MODERATE"
    overall_manipulation_score: float = 0.0  # 0-1
    recommendation: str = ""


class ManipulationDetector:
    """
    Detects institutional/bank manipulation patterns in precious metals.

    Patterns detected:
    - Stop hunts: Price spikes to hit stops then reverses
    - Liquidity sweeps: False breakouts at key levels
    - Volume anomalies: Unusual volume without news
    - Session manipulation: London open raids, NY reversals
    - Gap fills: Unfilled gaps that attract price
    - Order blocks: Institutional order zones
    """

    def __init__(self, lookback_periods: int = 100):
        """
        Initialize manipulation detector.

        Args:
            lookback_periods: Number of candles to analyze for patterns
        """
        self.lookback_periods = lookback_periods
        self._key_levels_cache: dict = {}
        self._order_blocks_cache: dict = {}

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _find_swing_points(
        self,
        df: pd.DataFrame,
        lookback: int = 5
    ) -> tuple[list[float], list[float]]:
        """Find swing highs and lows."""
        swing_highs = []
        swing_lows = []

        highs = df['high'].values
        lows = df['low'].values

        for i in range(lookback, len(df) - lookback):
            # Swing high: highest point in lookback window
            if highs[i] == max(highs[i - lookback:i + lookback + 1]):
                swing_highs.append(highs[i])

            # Swing low: lowest point in lookback window
            if lows[i] == min(lows[i - lookback:i + lookback + 1]):
                swing_lows.append(lows[i])

        return swing_highs, swing_lows

    def _find_key_levels(self, df: pd.DataFrame) -> list[KeyLevel]:
        """Identify key support/resistance levels."""
        if df.empty:
            return []

        swing_highs, swing_lows = self._find_swing_points(df)
        key_levels = []

        # Add swing highs as resistance
        for price in swing_highs[-10:]:  # Last 10 swing highs
            key_levels.append(KeyLevel(
                price=price,
                level_type="RESISTANCE",
                strength=0.7,
                touches=1
            ))

        # Add swing lows as support
        for price in swing_lows[-10:]:  # Last 10 swing lows
            key_levels.append(KeyLevel(
                price=price,
                level_type="SUPPORT",
                strength=0.7,
                touches=1
            ))

        # Add round numbers (psychological levels)
        current_price = df['close'].iloc[-1]
        round_levels = self._get_round_number_levels(current_price)
        for price in round_levels:
            key_levels.append(KeyLevel(
                price=price,
                level_type="ROUND_NUMBER",
                strength=0.5,
                touches=0
            ))

        # Consolidate nearby levels and count touches
        consolidated = self._consolidate_levels(key_levels, df)

        return sorted(consolidated, key=lambda x: x.price)

    def _get_round_number_levels(self, current_price: float) -> list[float]:
        """Get round number levels near current price."""
        # For gold, round to 10s and 50s
        if current_price > 1000:  # Gold-like prices
            step = 10
            major_step = 50
        else:  # Silver-like prices
            step = 0.5
            major_step = 1

        levels = []
        base = round(current_price / step) * step

        for i in range(-5, 6):
            level = base + (i * step)
            levels.append(level)

        return levels

    def _consolidate_levels(
        self,
        levels: list[KeyLevel],
        df: pd.DataFrame,
        tolerance: float = 0.002
    ) -> list[KeyLevel]:
        """Consolidate nearby levels and count price touches."""
        if not levels:
            return []

        consolidated = []
        used = set()

        for i, level in enumerate(levels):
            if i in used:
                continue

            # Find nearby levels
            nearby = [level]
            for j, other in enumerate(levels):
                if j != i and j not in used:
                    if level.price != 0 and abs(level.price - other.price) / level.price < tolerance:
                        nearby.append(other)
                        used.add(j)

            # Average the levels
            avg_price = np.mean([l.price for l in nearby])
            max_strength = max(l.strength for l in nearby)

            # Count touches in historical data
            touches = 0
            for idx, row in df.iterrows():
                if avg_price != 0 and abs(row['high'] - avg_price) / avg_price < tolerance:
                    touches += 1
                if avg_price != 0 and abs(row['low'] - avg_price) / avg_price < tolerance:
                    touches += 1

            consolidated.append(KeyLevel(
                price=avg_price,
                level_type=level.level_type,
                strength=min(1.0, max_strength + (touches * 0.1)),
                touches=touches
            ))

        return consolidated

    def _detect_stop_hunt(
        self,
        df: pd.DataFrame,
        key_levels: list[KeyLevel]
    ) -> list[ManipulationAlert]:
        """
        Detect stop hunt patterns.

        Pattern: Price spikes through a level, triggers stops, then reverses.
        """
        alerts = []

        if len(df) < 10:
            return alerts

        recent = df.tail(10)
        atr = self._calculate_atr(df).iloc[-1] if len(df) > 14 else df['high'].iloc[-1] - df['low'].iloc[-1]

        for level in key_levels:
            level_price = level.price

            # Check if price spiked through level and reversed
            for i in range(1, len(recent)):
                prev_candle = recent.iloc[i - 1]
                curr_candle = recent.iloc[i]

                # Bullish stop hunt (price dipped below support, then reversed up)
                if level.level_type == "SUPPORT":
                    if prev_candle['low'] < level_price < prev_candle['close']:
                        if curr_candle['close'] > level_price:
                            reversal_strength = (curr_candle['close'] - prev_candle['low']) / atr

                            if reversal_strength > 0.5:
                                alerts.append(ManipulationAlert(
                                    manipulation_type=ManipulationType.STOP_HUNT,
                                    severity=ManipulationSeverity.HIGH if reversal_strength > 1 else ManipulationSeverity.MODERATE,
                                    timestamp=curr_candle.name if hasattr(curr_candle, 'name') else datetime.now(timezone.utc),
                                    price_at_detection=curr_candle['close'],
                                    description=f"Bullish stop hunt detected at {level_price:.2f}",
                                    key_level_involved=level_price,
                                    expected_reversal="UP",
                                    confidence=min(0.9, 0.5 + reversal_strength * 0.2)
                                ))

                # Bearish stop hunt (price spiked above resistance, then reversed down)
                if level.level_type == "RESISTANCE":
                    if prev_candle['high'] > level_price > prev_candle['close']:
                        if curr_candle['close'] < level_price:
                            reversal_strength = (prev_candle['high'] - curr_candle['close']) / atr

                            if reversal_strength > 0.5:
                                alerts.append(ManipulationAlert(
                                    manipulation_type=ManipulationType.STOP_HUNT,
                                    severity=ManipulationSeverity.HIGH if reversal_strength > 1 else ManipulationSeverity.MODERATE,
                                    timestamp=curr_candle.name if hasattr(curr_candle, 'name') else datetime.now(timezone.utc),
                                    price_at_detection=curr_candle['close'],
                                    description=f"Bearish stop hunt detected at {level_price:.2f}",
                                    key_level_involved=level_price,
                                    expected_reversal="DOWN",
                                    confidence=min(0.9, 0.5 + reversal_strength * 0.2)
                                ))

        return alerts

    def _detect_liquidity_sweep(
        self,
        df: pd.DataFrame,
        key_levels: list[KeyLevel]
    ) -> list[ManipulationAlert]:
        """
        Detect liquidity sweep patterns.

        Pattern: Quick move through a level to grab liquidity, then reversal.
        """
        alerts = []

        if len(df) < 5:
            return alerts

        recent = df.tail(5)
        current = recent.iloc[-1]
        prev = recent.iloc[-2]

        for level in key_levels:
            level_price = level.price

            # Check for sweep below support (bullish sweep)
            if level.level_type == "SUPPORT":
                # Price swept below level
                if recent['low'].min() < level_price:
                    # But current close is above
                    if current['close'] > level_price:
                        sweep_depth = level_price - recent['low'].min()
                        recovery = current['close'] - recent['low'].min()

                        if recovery > sweep_depth * 1.5:
                            alerts.append(ManipulationAlert(
                                manipulation_type=ManipulationType.LIQUIDITY_SWEEP,
                                severity=ManipulationSeverity.HIGH,
                                timestamp=datetime.now(timezone.utc),
                                price_at_detection=current['close'],
                                description=f"Bullish liquidity sweep below {level_price:.2f}",
                                key_level_involved=level_price,
                                expected_reversal="UP",
                                confidence=0.75
                            ))

            # Check for sweep above resistance (bearish sweep)
            if level.level_type == "RESISTANCE":
                if recent['high'].max() > level_price:
                    if current['close'] < level_price:
                        sweep_height = recent['high'].max() - level_price
                        rejection = recent['high'].max() - current['close']

                        if rejection > sweep_height * 1.5:
                            alerts.append(ManipulationAlert(
                                manipulation_type=ManipulationType.LIQUIDITY_SWEEP,
                                severity=ManipulationSeverity.HIGH,
                                timestamp=datetime.now(timezone.utc),
                                price_at_detection=current['close'],
                                description=f"Bearish liquidity sweep above {level_price:.2f}",
                                key_level_involved=level_price,
                                expected_reversal="DOWN",
                                confidence=0.75
                            ))

        return alerts

    def _detect_volume_anomaly(self, df: pd.DataFrame) -> list[ManipulationAlert]:
        """
        Detect unusual volume spikes without corresponding price movement.
        """
        alerts = []

        if len(df) < 20 or 'volume' not in df.columns:
            return alerts

        # Calculate volume statistics
        vol = df['volume']
        avg_vol = vol.rolling(window=20).mean()
        std_vol = vol.rolling(window=20).std()

        recent = df.tail(5)

        for idx, row in recent.iterrows():
            vol_zscore = (row['volume'] - avg_vol.loc[idx]) / std_vol.loc[idx] if std_vol.loc[idx] > 0 else 0

            # Volume spike (more than 2 standard deviations)
            if vol_zscore > 2:
                # Check if price movement is proportional
                price_range = row['high'] - row['low']
                avg_range = (df['high'] - df['low']).rolling(window=20).mean().loc[idx]

                if avg_range > 0:
                    range_ratio = price_range / avg_range

                    # High volume but low price movement = suspicious
                    if range_ratio < 0.5:
                        alerts.append(ManipulationAlert(
                            manipulation_type=ManipulationType.VOLUME_ANOMALY,
                            severity=ManipulationSeverity.MODERATE,
                            timestamp=idx if isinstance(idx, datetime) else datetime.now(timezone.utc),
                            price_at_detection=row['close'],
                            description=f"Volume spike ({vol_zscore:.1f}x normal) with minimal price movement",
                            confidence=min(0.8, 0.5 + vol_zscore * 0.1)
                        ))

        return alerts

    def _detect_fake_breakout(
        self,
        df: pd.DataFrame,
        key_levels: list[KeyLevel]
    ) -> list[ManipulationAlert]:
        """
        Detect fake breakouts that quickly reverse.
        """
        alerts = []

        if len(df) < 10:
            return alerts

        recent = df.tail(10)

        for level in key_levels:
            level_price = level.price

            # Look for breakout and reversal pattern
            breakout_candle = None
            reversal_candle = None

            for i in range(len(recent) - 1):
                curr = recent.iloc[i]
                next_c = recent.iloc[i + 1]

                # Bullish breakout that failed
                if level.level_type == "RESISTANCE":
                    if curr['close'] > level_price > curr['open']:
                        # Check for reversal
                        if next_c['close'] < level_price:
                            alerts.append(ManipulationAlert(
                                manipulation_type=ManipulationType.FAKE_BREAKOUT,
                                severity=ManipulationSeverity.HIGH,
                                timestamp=next_c.name if hasattr(next_c, 'name') else datetime.now(timezone.utc),
                                price_at_detection=next_c['close'],
                                description=f"Fake bullish breakout above {level_price:.2f}",
                                key_level_involved=level_price,
                                expected_reversal="DOWN",
                                confidence=0.70
                            ))

                # Bearish breakout that failed
                if level.level_type == "SUPPORT":
                    if curr['close'] < level_price < curr['open']:
                        if next_c['close'] > level_price:
                            alerts.append(ManipulationAlert(
                                manipulation_type=ManipulationType.FAKE_BREAKOUT,
                                severity=ManipulationSeverity.HIGH,
                                timestamp=next_c.name if hasattr(next_c, 'name') else datetime.now(timezone.utc),
                                price_at_detection=next_c['close'],
                                description=f"Fake bearish breakout below {level_price:.2f}",
                                key_level_involved=level_price,
                                expected_reversal="UP",
                                confidence=0.70
                            ))

        return alerts

    def _detect_order_blocks(self, df: pd.DataFrame) -> list[OrderBlock]:
        """
        Detect institutional order blocks.

        Order blocks are zones where institutions placed large orders,
        often marked by strong moves away from a consolidation zone.
        """
        order_blocks = []

        if len(df) < 20:
            return order_blocks

        atr = self._calculate_atr(df, period=14)

        for i in range(5, len(df) - 1):
            curr = df.iloc[i]
            prev = df.iloc[i - 1]
            avg_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else (curr['high'] - curr['low'])

            # Bullish order block: bearish candle followed by strong bullish move
            if prev['close'] < prev['open']:  # Previous was bearish
                if curr['close'] > curr['open']:  # Current is bullish
                    move = curr['close'] - prev['low']
                    if move > avg_atr * 1.5:  # Strong move up
                        order_blocks.append(OrderBlock(
                            price_high=prev['open'],
                            price_low=prev['low'],
                            block_type="BULLISH",
                            strength=min(1.0, move / (avg_atr * 2)),
                            timestamp=prev.name if hasattr(prev, 'name') else datetime.now(timezone.utc)
                        ))

            # Bearish order block: bullish candle followed by strong bearish move
            if prev['close'] > prev['open']:  # Previous was bullish
                if curr['close'] < curr['open']:  # Current is bearish
                    move = prev['high'] - curr['close']
                    if move > avg_atr * 1.5:  # Strong move down
                        order_blocks.append(OrderBlock(
                            price_high=prev['high'],
                            price_low=prev['open'],
                            block_type="BEARISH",
                            strength=min(1.0, move / (avg_atr * 2)),
                            timestamp=prev.name if hasattr(prev, 'name') else datetime.now(timezone.utc)
                        ))

        # Check if order blocks have been tested or broken
        current_price = df['close'].iloc[-1]
        for ob in order_blocks:
            if ob.block_type == "BULLISH":
                if current_price < ob.price_low:
                    ob.is_broken = True
                elif ob.price_low <= current_price <= ob.price_high:
                    ob.is_tested = True
            else:  # BEARISH
                if current_price > ob.price_high:
                    ob.is_broken = True
                elif ob.price_low <= current_price <= ob.price_high:
                    ob.is_tested = True

        # Return most recent valid order blocks
        valid_blocks = [ob for ob in order_blocks if not ob.is_broken]
        return sorted(valid_blocks, key=lambda x: x.strength, reverse=True)[:5]

    def _get_session_risk(self, current_hour: int) -> tuple[str, str]:
        """Get manipulation risk based on current trading session."""
        session_risks = {
            (0, 8): ("ASIAN", "LOW"),
            (8, 13): ("LONDON", "HIGH"),
            (13, 14): ("NY_OVERLAP", "HIGHEST"),
            (14, 20): ("NEW_YORK", "MODERATE"),
            (20, 24): ("NY_CLOSE", "MODERATE"),
        }

        for (start, end), (session, risk) in session_risks.items():
            if start <= current_hour < end:
                return session, risk

        return "UNKNOWN", "MODERATE"

    def _calculate_manipulation_score(
        self,
        alerts: list[ManipulationAlert],
        session_risk: str
    ) -> float:
        """Calculate overall manipulation score (0-1)."""
        score = 0.0

        # Base score from session risk
        session_scores = {
            "LOW": 0.1,
            "MODERATE": 0.3,
            "HIGH": 0.5,
            "HIGHEST": 0.7,
        }
        score += session_scores.get(session_risk, 0.3)

        # Add score from active alerts
        severity_scores = {
            ManipulationSeverity.LOW: 0.05,
            ManipulationSeverity.MODERATE: 0.1,
            ManipulationSeverity.HIGH: 0.15,
            ManipulationSeverity.CRITICAL: 0.2,
        }

        for alert in alerts:
            score += severity_scores.get(alert.severity, 0.1)

        return min(1.0, score)

    def _generate_recommendation(
        self,
        manipulation_score: float,
        alerts: list[ManipulationAlert],
        session_risk: str
    ) -> str:
        """Generate trading recommendation based on manipulation analysis."""
        if manipulation_score > 0.7:
            return "AVOID TRADING - High manipulation activity detected. Wait for cleaner price action."

        if manipulation_score > 0.5:
            if alerts:
                reversal_directions = [a.expected_reversal for a in alerts if a.expected_reversal]
                if reversal_directions:
                    common_direction = max(set(reversal_directions), key=reversal_directions.count)
                    return f"CAUTION - Manipulation detected. Potential {common_direction} reversal. Trade with tight stops."

            return "CAUTION - Elevated manipulation risk. Reduce position size."

        if session_risk in ["HIGH", "HIGHEST"]:
            return "BE AWARE - High manipulation session. Watch for stop hunts at key levels."

        return "NORMAL - No significant manipulation detected. Trade with standard risk management."

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str = "XAU/USD"
    ) -> ManipulationAnalysis:
        """
        Perform complete manipulation analysis.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            symbol: Asset symbol

        Returns:
            ManipulationAnalysis with all detected patterns
        """
        if df.empty:
            return ManipulationAnalysis(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                current_price=0,
                recommendation="No data available for analysis"
            )

        current_price = df['close'].iloc[-1]
        current_hour = datetime.now(timezone.utc).hour

        # Find key levels
        key_levels = self._find_key_levels(df)

        # Detect various manipulation patterns
        alerts = []
        alerts.extend(self._detect_stop_hunt(df, key_levels))
        alerts.extend(self._detect_liquidity_sweep(df, key_levels))
        alerts.extend(self._detect_volume_anomaly(df))
        alerts.extend(self._detect_fake_breakout(df, key_levels))

        # Detect order blocks
        order_blocks = self._detect_order_blocks(df)

        # Get session risk
        session_name, session_risk = self._get_session_risk(current_hour)

        # Calculate overall score
        manipulation_score = self._calculate_manipulation_score(alerts, session_risk)

        # Generate recommendation
        recommendation = self._generate_recommendation(manipulation_score, alerts, session_risk)

        # Sort alerts by severity
        alerts.sort(key=lambda x: x.severity.value, reverse=True)

        return ManipulationAnalysis(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            current_price=current_price,
            active_alerts=alerts[:5],  # Top 5 alerts
            key_levels=key_levels[:10],  # Top 10 levels
            order_blocks=order_blocks,
            session_risk=session_risk,
            overall_manipulation_score=manipulation_score,
            recommendation=recommendation
        )

    def get_quick_status(self, df: pd.DataFrame) -> dict:
        """Get a quick manipulation status for dashboard display."""
        analysis = self.analyze(df)

        return {
            "manipulation_score": analysis.overall_manipulation_score,
            "risk_level": "HIGH" if analysis.overall_manipulation_score > 0.6 else
                         "MODERATE" if analysis.overall_manipulation_score > 0.3 else "LOW",
            "session_risk": analysis.session_risk,
            "active_alerts_count": len(analysis.active_alerts),
            "recommendation": analysis.recommendation,
            "key_levels": [
                {"price": l.price, "type": l.level_type, "strength": l.strength}
                for l in analysis.key_levels[:5]
            ]
        }


# Singleton instance
manipulation_detector = ManipulationDetector()
