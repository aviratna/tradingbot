"""
Trade Signal Generator for Precious Metals Trading.
Generates entry/exit signals combining technical analysis, divergences, and correlations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Trade signal direction."""
    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"


class SignalStrength(Enum):
    """Signal strength classification."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class SignalType(Enum):
    """Type of signal trigger."""
    FIBONACCI_BOUNCE = "fibonacci_bounce"
    RSI_DIVERGENCE = "rsi_divergence"
    MACD_CROSSOVER = "macd_crossover"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    CORRELATION_BASED = "correlation_based"
    POST_MANIPULATION = "post_manipulation"
    MULTI_TIMEFRAME = "multi_timeframe"


@dataclass
class TechnicalIndicators:
    """Current technical indicator values."""
    rsi: float = 50.0
    rsi_prev: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    atr: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    bollinger_middle: float = 0.0
    stochastic_k: float = 50.0
    stochastic_d: float = 50.0


@dataclass
class DivergenceSignal:
    """Detected divergence between price and indicator."""
    divergence_type: str  # BULLISH, BEARISH, HIDDEN_BULLISH, HIDDEN_BEARISH
    indicator: str  # RSI, MACD, STOCHASTIC
    strength: float
    price_trend: str  # UP, DOWN
    indicator_trend: str  # UP, DOWN
    confidence: float


@dataclass
class CorrelationSignal:
    """Signal based on correlated asset movements."""
    correlated_asset: str
    correlation_type: str  # POSITIVE, NEGATIVE, INVERSE
    asset_movement: str  # UP, DOWN
    expected_metal_movement: str  # UP, DOWN
    alignment_score: float  # How aligned is current movement with expectation


@dataclass
class TradeSignal:
    """Complete trade signal with entry, stop loss, and take profit."""
    asset: str
    direction: SignalDirection
    signal_type: SignalType
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    position_size_lots: float
    risk_amount: float
    confidence: float
    strength: SignalStrength
    expiry_minutes: int
    reasoning: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    indicators: Optional[TechnicalIndicators] = None
    divergence: Optional[DivergenceSignal] = None
    correlations: list[CorrelationSignal] = field(default_factory=list)

    @property
    def risk_reward_1(self) -> float:
        """Calculate risk:reward ratio for TP1."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit_1 - self.entry_price)
        return reward / risk if risk > 0 else 0

    @property
    def risk_reward_2(self) -> float:
        """Calculate risk:reward ratio for TP2."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit_2 - self.entry_price)
        return reward / risk if risk > 0 else 0


@dataclass
class SignalAnalysis:
    """Complete signal analysis result."""
    symbol: str
    timestamp: datetime
    current_price: float
    active_signals: list[TradeSignal]
    indicators: TechnicalIndicators
    divergences: list[DivergenceSignal]
    market_bias: str  # BULLISH, BEARISH, NEUTRAL
    overall_confidence: float


class SignalGenerator:
    """
    Generates trade signals for precious metals using multiple indicators.

    Signal Types:
    - Fibonacci bounces at key retracement levels
    - RSI/MACD divergences
    - Breakout signals with volume confirmation
    - Post-manipulation reversal signals
    - Correlation-based signals (DXY, Yields, etc.)
    """

    # Fibonacci levels for signal generation
    FIBONACCI_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
    KEY_FIBONACCI = [0.382, 0.618]  # Strongest levels

    # Default risk parameters
    DEFAULT_RISK_PERCENT = 2.0  # 2% account risk per trade
    DEFAULT_ACCOUNT_SIZE = 5000  # $5000 default account

    def __init__(
        self,
        account_size: float = 5000,
        risk_percent: float = 2.0,
        min_rr_ratio: float = 1.5
    ):
        """
        Initialize signal generator.

        Args:
            account_size: Trading account size in USD
            risk_percent: Percentage of account to risk per trade
            min_rr_ratio: Minimum risk:reward ratio for signals
        """
        self.account_size = account_size
        self.risk_percent = risk_percent
        self.min_rr_ratio = min_rr_ratio

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate EMA."""
        return prices.ewm(span=period, adjust=False).mean()

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

    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def _calculate_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator."""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()

        k = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()

        return k, d

    def _get_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Calculate all technical indicators."""
        if df.empty or len(df) < 30:
            return TechnicalIndicators()

        close = df['close']

        # RSI
        rsi = self._calculate_rsi(close)
        rsi_current = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        rsi_prev = rsi.iloc[-2] if len(rsi) > 1 and not pd.isna(rsi.iloc[-2]) else 50

        # MACD
        macd, macd_signal, macd_histogram = self._calculate_macd(close)

        # EMAs
        ema_9 = self._calculate_ema(close, 9)
        ema_21 = self._calculate_ema(close, 21)
        ema_50 = self._calculate_ema(close, 50)

        # ATR
        atr = self._calculate_atr(df)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close)

        # Stochastic
        stoch_k, stoch_d = self._calculate_stochastic(df)

        return TechnicalIndicators(
            rsi=rsi_current,
            rsi_prev=rsi_prev,
            macd=macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0,
            macd_signal=macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else 0,
            macd_histogram=macd_histogram.iloc[-1] if not pd.isna(macd_histogram.iloc[-1]) else 0,
            ema_9=ema_9.iloc[-1] if not pd.isna(ema_9.iloc[-1]) else 0,
            ema_21=ema_21.iloc[-1] if not pd.isna(ema_21.iloc[-1]) else 0,
            ema_50=ema_50.iloc[-1] if not pd.isna(ema_50.iloc[-1]) else 0,
            atr=atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0,
            bollinger_upper=bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else 0,
            bollinger_lower=bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else 0,
            bollinger_middle=bb_middle.iloc[-1] if not pd.isna(bb_middle.iloc[-1]) else 0,
            stochastic_k=stoch_k.iloc[-1] if not pd.isna(stoch_k.iloc[-1]) else 50,
            stochastic_d=stoch_d.iloc[-1] if not pd.isna(stoch_d.iloc[-1]) else 50
        )

    def _detect_divergences(
        self,
        df: pd.DataFrame,
        lookback: int = 20
    ) -> list[DivergenceSignal]:
        """Detect divergences between price and indicators."""
        divergences = []

        if len(df) < lookback:
            return divergences

        close = df['close']
        rsi = self._calculate_rsi(close)

        recent_close = close.tail(lookback)
        recent_rsi = rsi.tail(lookback)

        # Find price highs and lows
        price_high_idx = recent_close.idxmax()
        price_low_idx = recent_close.idxmin()

        # Get corresponding RSI values
        rsi_at_price_high = recent_rsi.loc[price_high_idx] if price_high_idx in recent_rsi.index else None
        rsi_at_price_low = recent_rsi.loc[price_low_idx] if price_low_idx in recent_rsi.index else None

        # Compare recent highs/lows with previous
        mid_point = lookback // 2

        if len(recent_close) > mid_point:
            # Price making higher high but RSI making lower high = bearish divergence
            first_half_high = recent_close.iloc[:mid_point].max()
            second_half_high = recent_close.iloc[mid_point:].max()
            first_half_rsi_high = recent_rsi.iloc[:mid_point].max()
            second_half_rsi_high = recent_rsi.iloc[mid_point:].max()

            if second_half_high > first_half_high and second_half_rsi_high < first_half_rsi_high:
                if second_half_rsi_high > 50:  # Confirm overbought area
                    divergences.append(DivergenceSignal(
                        divergence_type="BEARISH",
                        indicator="RSI",
                        strength=0.7,
                        price_trend="UP",
                        indicator_trend="DOWN",
                        confidence=0.65
                    ))

            # Price making lower low but RSI making higher low = bullish divergence
            first_half_low = recent_close.iloc[:mid_point].min()
            second_half_low = recent_close.iloc[mid_point:].min()
            first_half_rsi_low = recent_rsi.iloc[:mid_point].min()
            second_half_rsi_low = recent_rsi.iloc[mid_point:].min()

            if second_half_low < first_half_low and second_half_rsi_low > first_half_rsi_low:
                if second_half_rsi_low < 50:  # Confirm oversold area
                    divergences.append(DivergenceSignal(
                        divergence_type="BULLISH",
                        indicator="RSI",
                        strength=0.7,
                        price_trend="DOWN",
                        indicator_trend="UP",
                        confidence=0.65
                    ))

        return divergences

    def _find_fibonacci_levels(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> dict[str, list[float]]:
        """Calculate Fibonacci retracement levels."""
        if len(df) < lookback:
            return {"retracements": [], "extensions": []}

        recent = df.tail(lookback)
        high = recent['high'].max()
        low = recent['low'].min()
        diff = high - low

        # Determine trend
        first_close = recent['close'].iloc[0]
        last_close = recent['close'].iloc[-1]
        is_uptrend = last_close > first_close

        retracements = []
        extensions = []

        for level in self.FIBONACCI_LEVELS:
            if is_uptrend:
                # Retracements from high
                retrace_price = high - (diff * level)
                retracements.append(retrace_price)
            else:
                # Retracements from low
                retrace_price = low + (diff * level)
                retracements.append(retrace_price)

        # Extensions
        if is_uptrend:
            extensions = [high + (diff * ext) for ext in [0.618, 1.0, 1.618]]
        else:
            extensions = [low - (diff * ext) for ext in [0.618, 1.0, 1.618]]

        return {
            "retracements": retracements,
            "extensions": extensions,
            "is_uptrend": is_uptrend,
            "swing_high": high,
            "swing_low": low
        }

    def _check_fibonacci_signal(
        self,
        current_price: float,
        fib_data: dict,
        indicators: TechnicalIndicators
    ) -> Optional[TradeSignal]:
        """Check for Fibonacci bounce signals."""
        if not fib_data.get("retracements"):
            return None

        tolerance = 0.001  # 0.1% tolerance

        for i, level_price in enumerate(fib_data["retracements"]):
            # Check if price is near a Fibonacci level
            if abs(current_price - level_price) / current_price < tolerance:
                level = self.FIBONACCI_LEVELS[i] if i < len(self.FIBONACCI_LEVELS) else 0.5

                # Confirm with RSI
                if fib_data.get("is_uptrend", True):
                    # Looking for buy at retracement
                    if indicators.rsi < 40 and level in self.KEY_FIBONACCI:
                        return self._create_signal(
                            direction=SignalDirection.BUY,
                            signal_type=SignalType.FIBONACCI_BOUNCE,
                            current_price=current_price,
                            atr=indicators.atr,
                            reasoning=f"Price at {level*100:.1f}% Fib retracement with RSI oversold ({indicators.rsi:.1f})",
                            confidence=0.7 if level == 0.618 else 0.6
                        )
                else:
                    # Looking for sell at retracement
                    if indicators.rsi > 60 and level in self.KEY_FIBONACCI:
                        return self._create_signal(
                            direction=SignalDirection.SELL,
                            signal_type=SignalType.FIBONACCI_BOUNCE,
                            current_price=current_price,
                            atr=indicators.atr,
                            reasoning=f"Price at {level*100:.1f}% Fib retracement with RSI overbought ({indicators.rsi:.1f})",
                            confidence=0.7 if level == 0.618 else 0.6
                        )

        return None

    def _check_divergence_signal(
        self,
        divergences: list[DivergenceSignal],
        current_price: float,
        atr: float
    ) -> Optional[TradeSignal]:
        """Generate signal from divergence."""
        if not divergences:
            return None

        # Use the strongest divergence
        strongest = max(divergences, key=lambda x: x.confidence)

        if strongest.divergence_type == "BULLISH":
            return self._create_signal(
                direction=SignalDirection.BUY,
                signal_type=SignalType.RSI_DIVERGENCE,
                current_price=current_price,
                atr=atr,
                reasoning=f"Bullish {strongest.indicator} divergence detected",
                confidence=strongest.confidence
            )
        elif strongest.divergence_type == "BEARISH":
            return self._create_signal(
                direction=SignalDirection.SELL,
                signal_type=SignalType.RSI_DIVERGENCE,
                current_price=current_price,
                atr=atr,
                reasoning=f"Bearish {strongest.indicator} divergence detected",
                confidence=strongest.confidence
            )

        return None

    def _check_macd_signal(
        self,
        indicators: TechnicalIndicators,
        current_price: float
    ) -> Optional[TradeSignal]:
        """Check for MACD crossover signals."""
        # MACD crossing above signal line
        if indicators.macd > indicators.macd_signal:
            if indicators.macd_histogram > 0:
                # Confirm with RSI not overbought
                if indicators.rsi < 70:
                    return self._create_signal(
                        direction=SignalDirection.BUY,
                        signal_type=SignalType.MACD_CROSSOVER,
                        current_price=current_price,
                        atr=indicators.atr,
                        reasoning=f"MACD bullish crossover with histogram positive",
                        confidence=0.55
                    )

        # MACD crossing below signal line
        if indicators.macd < indicators.macd_signal:
            if indicators.macd_histogram < 0:
                # Confirm with RSI not oversold
                if indicators.rsi > 30:
                    return self._create_signal(
                        direction=SignalDirection.SELL,
                        signal_type=SignalType.MACD_CROSSOVER,
                        current_price=current_price,
                        atr=indicators.atr,
                        reasoning=f"MACD bearish crossover with histogram negative",
                        confidence=0.55
                    )

        return None

    def _check_ema_signal(
        self,
        indicators: TechnicalIndicators,
        current_price: float
    ) -> Optional[TradeSignal]:
        """Check for EMA crossover signals."""
        # Fast EMA above slow = bullish
        if indicators.ema_9 > indicators.ema_21 > indicators.ema_50:
            if current_price > indicators.ema_9:
                if indicators.rsi < 65:  # Not overbought
                    return self._create_signal(
                        direction=SignalDirection.BUY,
                        signal_type=SignalType.BREAKOUT,
                        current_price=current_price,
                        atr=indicators.atr,
                        reasoning="Price above aligned bullish EMAs (9 > 21 > 50)",
                        confidence=0.60
                    )

        # Fast EMA below slow = bearish
        if indicators.ema_9 < indicators.ema_21 < indicators.ema_50:
            if current_price < indicators.ema_9:
                if indicators.rsi > 35:  # Not oversold
                    return self._create_signal(
                        direction=SignalDirection.SELL,
                        signal_type=SignalType.BREAKOUT,
                        current_price=current_price,
                        atr=indicators.atr,
                        reasoning="Price below aligned bearish EMAs (9 < 21 < 50)",
                        confidence=0.60
                    )

        return None

    def _check_reversal_signal(
        self,
        df: pd.DataFrame,
        indicators: TechnicalIndicators,
        current_price: float
    ) -> Optional[TradeSignal]:
        """Check for reversal signals at extremes."""
        # Oversold reversal
        if indicators.rsi < 25:
            if indicators.stochastic_k < 20 and indicators.stochastic_k > indicators.stochastic_d:
                # Check for bullish candle
                if len(df) > 1:
                    last_candle = df.iloc[-1]
                    if last_candle['close'] > last_candle['open']:
                        return self._create_signal(
                            direction=SignalDirection.BUY,
                            signal_type=SignalType.REVERSAL,
                            current_price=current_price,
                            atr=indicators.atr,
                            reasoning=f"Oversold reversal: RSI={indicators.rsi:.1f}, Stoch turning up",
                            confidence=0.70
                        )

        # Overbought reversal
        if indicators.rsi > 75:
            if indicators.stochastic_k > 80 and indicators.stochastic_k < indicators.stochastic_d:
                # Check for bearish candle
                if len(df) > 1:
                    last_candle = df.iloc[-1]
                    if last_candle['close'] < last_candle['open']:
                        return self._create_signal(
                            direction=SignalDirection.SELL,
                            signal_type=SignalType.REVERSAL,
                            current_price=current_price,
                            atr=indicators.atr,
                            reasoning=f"Overbought reversal: RSI={indicators.rsi:.1f}, Stoch turning down",
                            confidence=0.70
                        )

        return None

    def _create_signal(
        self,
        direction: SignalDirection,
        signal_type: SignalType,
        current_price: float,
        atr: float,
        reasoning: str,
        confidence: float,
        symbol: str = "XAU/USD"
    ) -> TradeSignal:
        """Create a complete trade signal with risk management."""
        # Calculate stop loss and take profits based on ATR
        if atr <= 0:
            atr = current_price * 0.002  # Default 0.2% of price

        # For gold, typical pip value is $10 per 0.01 lot per pip
        pip_value = 10 if "XAU" in symbol else 5  # Silver has different pip value

        if direction == SignalDirection.BUY:
            stop_loss = current_price - (atr * 1.5)
            take_profit_1 = current_price + (atr * 2)
            take_profit_2 = current_price + (atr * 3.5)
        else:
            stop_loss = current_price + (atr * 1.5)
            take_profit_1 = current_price - (atr * 2)
            take_profit_2 = current_price - (atr * 3.5)

        # Calculate position size
        risk_amount = self.account_size * (self.risk_percent / 100)
        stop_pips = abs(current_price - stop_loss) / 0.01  # Convert to pips
        position_size = risk_amount / (stop_pips * pip_value) if stop_pips > 0 else 0.01
        position_size = round(position_size, 2)
        position_size = max(0.01, min(position_size, 1.0))  # Limit between 0.01 and 1.0 lots

        # Determine signal strength
        if confidence >= 0.75:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.65:
            strength = SignalStrength.STRONG
        elif confidence >= 0.55:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        return TradeSignal(
            asset=symbol,
            direction=direction,
            signal_type=signal_type,
            entry_price=current_price,
            stop_loss=round(stop_loss, 2),
            take_profit_1=round(take_profit_1, 2),
            take_profit_2=round(take_profit_2, 2),
            position_size_lots=position_size,
            risk_amount=round(risk_amount, 2),
            confidence=round(confidence, 2),
            strength=strength,
            expiry_minutes=15,
            reasoning=reasoning
        )

    def _determine_market_bias(self, indicators: TechnicalIndicators) -> str:
        """Determine overall market bias."""
        bullish_score = 0
        bearish_score = 0

        # RSI
        if indicators.rsi > 50:
            bullish_score += 1
        else:
            bearish_score += 1

        # MACD
        if indicators.macd > indicators.macd_signal:
            bullish_score += 1
        else:
            bearish_score += 1

        # EMA alignment
        if indicators.ema_9 > indicators.ema_21:
            bullish_score += 1
        else:
            bearish_score += 1

        if indicators.ema_21 > indicators.ema_50:
            bullish_score += 1
        else:
            bearish_score += 1

        if bullish_score > bearish_score + 1:
            return "BULLISH"
        elif bearish_score > bullish_score + 1:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str = "XAU/USD",
        manipulation_alerts: Optional[list] = None
    ) -> SignalAnalysis:
        """
        Generate trade signals from price data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Asset symbol
            manipulation_alerts: Optional list of manipulation alerts to consider

        Returns:
            SignalAnalysis with all detected signals
        """
        if df.empty or len(df) < 30:
            return SignalAnalysis(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                current_price=0,
                active_signals=[],
                indicators=TechnicalIndicators(),
                divergences=[],
                market_bias="NEUTRAL",
                overall_confidence=0
            )

        current_price = df['close'].iloc[-1]

        # Calculate indicators
        indicators = self._get_indicators(df)

        # Detect divergences
        divergences = self._detect_divergences(df)

        # Find Fibonacci levels
        fib_data = self._find_fibonacci_levels(df)

        # Collect signals
        signals = []

        # Check various signal types
        fib_signal = self._check_fibonacci_signal(current_price, fib_data, indicators)
        if fib_signal:
            fib_signal.asset = symbol
            signals.append(fib_signal)

        div_signal = self._check_divergence_signal(divergences, current_price, indicators.atr)
        if div_signal:
            div_signal.asset = symbol
            div_signal.divergence = divergences[0] if divergences else None
            signals.append(div_signal)

        macd_signal = self._check_macd_signal(indicators, current_price)
        if macd_signal:
            macd_signal.asset = symbol
            signals.append(macd_signal)

        ema_signal = self._check_ema_signal(indicators, current_price)
        if ema_signal:
            ema_signal.asset = symbol
            signals.append(ema_signal)

        reversal_signal = self._check_reversal_signal(df, indicators, current_price)
        if reversal_signal:
            reversal_signal.asset = symbol
            signals.append(reversal_signal)

        # Check for post-manipulation signals
        if manipulation_alerts:
            for alert in manipulation_alerts:
                if hasattr(alert, 'expected_reversal') and alert.expected_reversal:
                    direction = SignalDirection.BUY if alert.expected_reversal == "UP" else SignalDirection.SELL
                    manip_signal = self._create_signal(
                        direction=direction,
                        signal_type=SignalType.POST_MANIPULATION,
                        current_price=current_price,
                        atr=indicators.atr,
                        reasoning=f"Post-manipulation reversal expected after {alert.description}",
                        confidence=alert.confidence if hasattr(alert, 'confidence') else 0.6,
                        symbol=symbol
                    )
                    signals.append(manip_signal)

        # Add indicators to all signals
        for signal in signals:
            signal.indicators = indicators

        # Determine market bias
        market_bias = self._determine_market_bias(indicators)

        # Calculate overall confidence
        overall_confidence = np.mean([s.confidence for s in signals]) if signals else 0

        # Sort signals by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)

        return SignalAnalysis(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            current_price=current_price,
            active_signals=signals[:3],  # Top 3 signals
            indicators=indicators,
            divergences=divergences,
            market_bias=market_bias,
            overall_confidence=overall_confidence
        )

    def get_signal_summary(self, df: pd.DataFrame, symbol: str = "XAU/USD") -> dict:
        """Get a quick signal summary for dashboard display."""
        analysis = self.generate_signals(df, symbol)

        top_signal = analysis.active_signals[0] if analysis.active_signals else None

        return {
            "has_signal": len(analysis.active_signals) > 0,
            "market_bias": analysis.market_bias,
            "overall_confidence": analysis.overall_confidence,
            "top_signal": {
                "direction": top_signal.direction.value if top_signal else None,
                "entry": top_signal.entry_price if top_signal else None,
                "stop_loss": top_signal.stop_loss if top_signal else None,
                "take_profit_1": top_signal.take_profit_1 if top_signal else None,
                "take_profit_2": top_signal.take_profit_2 if top_signal else None,
                "position_size": top_signal.position_size_lots if top_signal else None,
                "confidence": top_signal.confidence if top_signal else None,
                "reasoning": top_signal.reasoning if top_signal else None,
                "risk_reward_1": top_signal.risk_reward_1 if top_signal else None,
                "risk_reward_2": top_signal.risk_reward_2 if top_signal else None,
            } if top_signal else None,
            "indicators": {
                "rsi": analysis.indicators.rsi,
                "macd": analysis.indicators.macd,
                "macd_signal": analysis.indicators.macd_signal,
                "ema_9": analysis.indicators.ema_9,
                "ema_21": analysis.indicators.ema_21,
            },
            "signal_count": len(analysis.active_signals)
        }


# Singleton instance
signal_generator = SignalGenerator()
