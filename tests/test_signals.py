"""Tests for the trade signal generator."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from app.analysis.signals import (
    SignalGenerator,
    SignalDirection,
    SignalStrength,
    SignalType,
    TechnicalIndicators,
    TradeSignal,
    SignalAnalysis
)


@pytest.fixture
def signal_generator():
    """Create a signal generator instance."""
    return SignalGenerator(
        account_size=5000,
        risk_percent=2.0,
        min_rr_ratio=1.5
    )


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')

    # Generate realistic gold price data around $2000
    base_price = 2000
    np.random.seed(42)

    prices = [base_price]
    for i in range(99):
        change = np.random.normal(0, 2)  # Random walk
        prices.append(prices[-1] + change)

    prices = np.array(prices)

    df = pd.DataFrame({
        'open': prices - np.random.uniform(0, 1, 100),
        'high': prices + np.random.uniform(0, 3, 100),
        'low': prices - np.random.uniform(0, 3, 100),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    return df


@pytest.fixture
def bullish_ohlcv_df():
    """Create bullish trending OHLCV data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')

    # Generate upward trending data
    base_price = 2000
    trend = np.linspace(0, 50, 100)  # 50 point upward trend
    noise = np.random.normal(0, 1, 100)
    prices = base_price + trend + noise

    df = pd.DataFrame({
        'open': prices - np.random.uniform(0, 1, 100),
        'high': prices + np.random.uniform(0, 3, 100),
        'low': prices - np.random.uniform(0, 3, 100),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    return df


@pytest.fixture
def bearish_ohlcv_df():
    """Create bearish trending OHLCV data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')

    # Generate downward trending data
    base_price = 2050
    trend = np.linspace(0, -50, 100)  # 50 point downward trend
    noise = np.random.normal(0, 1, 100)
    prices = base_price + trend + noise

    df = pd.DataFrame({
        'open': prices - np.random.uniform(0, 1, 100),
        'high': prices + np.random.uniform(0, 3, 100),
        'low': prices - np.random.uniform(0, 3, 100),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    return df


class TestSignalGenerator:
    """Tests for SignalGenerator class."""

    def test_initialization(self, signal_generator):
        """Test signal generator initialization."""
        assert signal_generator.account_size == 5000
        assert signal_generator.risk_percent == 2.0
        assert signal_generator.min_rr_ratio == 1.5

    def test_rsi_calculation(self, signal_generator, sample_ohlcv_df):
        """Test RSI calculation."""
        close = sample_ohlcv_df['close']
        rsi = signal_generator._calculate_rsi(close)

        assert len(rsi) == len(close)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_macd_calculation(self, signal_generator, sample_ohlcv_df):
        """Test MACD calculation."""
        close = sample_ohlcv_df['close']
        macd, signal, histogram = signal_generator._calculate_macd(close)

        assert len(macd) == len(close)
        assert len(signal) == len(close)
        assert len(histogram) == len(close)

        # Histogram should equal MACD - Signal
        valid_idx = ~(macd.isna() | signal.isna())
        np.testing.assert_array_almost_equal(
            histogram[valid_idx].values,
            (macd[valid_idx] - signal[valid_idx]).values,
            decimal=10
        )

    def test_ema_calculation(self, signal_generator, sample_ohlcv_df):
        """Test EMA calculation."""
        close = sample_ohlcv_df['close']

        ema_9 = signal_generator._calculate_ema(close, 9)
        ema_21 = signal_generator._calculate_ema(close, 21)

        assert len(ema_9) == len(close)
        assert len(ema_21) == len(close)

        # EMA should smooth the data
        assert ema_9.std() < close.std()

    def test_atr_calculation(self, signal_generator, sample_ohlcv_df):
        """Test ATR calculation."""
        atr = signal_generator._calculate_atr(sample_ohlcv_df)

        assert len(atr) == len(sample_ohlcv_df)
        # ATR should always be positive
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()

    def test_bollinger_bands(self, signal_generator, sample_ohlcv_df):
        """Test Bollinger Bands calculation."""
        close = sample_ohlcv_df['close']
        upper, middle, lower = signal_generator._calculate_bollinger_bands(close)

        assert len(upper) == len(close)
        assert len(middle) == len(close)
        assert len(lower) == len(close)

        # Upper should be above middle, lower should be below
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (lower[valid_idx] <= middle[valid_idx]).all()

    def test_stochastic_calculation(self, signal_generator, sample_ohlcv_df):
        """Test Stochastic oscillator calculation."""
        k, d = signal_generator._calculate_stochastic(sample_ohlcv_df)

        assert len(k) == len(sample_ohlcv_df)
        assert len(d) == len(sample_ohlcv_df)

        # Stochastic should be between 0 and 100
        valid_k = k.dropna()
        valid_d = d.dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()

    def test_get_indicators(self, signal_generator, sample_ohlcv_df):
        """Test getting all technical indicators."""
        indicators = signal_generator._get_indicators(sample_ohlcv_df)

        assert isinstance(indicators, TechnicalIndicators)
        assert 0 <= indicators.rsi <= 100
        assert indicators.atr >= 0
        assert 0 <= indicators.stochastic_k <= 100
        assert 0 <= indicators.stochastic_d <= 100

    def test_detect_divergences(self, signal_generator, sample_ohlcv_df):
        """Test divergence detection."""
        divergences = signal_generator._detect_divergences(sample_ohlcv_df)

        assert isinstance(divergences, list)
        for div in divergences:
            assert div.divergence_type in ["BULLISH", "BEARISH", "HIDDEN_BULLISH", "HIDDEN_BEARISH"]
            assert div.indicator in ["RSI", "MACD", "STOCHASTIC"]
            assert 0 <= div.confidence <= 1

    def test_find_fibonacci_levels(self, signal_generator, sample_ohlcv_df):
        """Test Fibonacci level calculation."""
        fib_data = signal_generator._find_fibonacci_levels(sample_ohlcv_df)

        assert "retracements" in fib_data
        assert "extensions" in fib_data
        assert "is_uptrend" in fib_data
        assert len(fib_data["retracements"]) > 0

    def test_generate_signals_returns_analysis(self, signal_generator, sample_ohlcv_df):
        """Test that generate_signals returns SignalAnalysis."""
        analysis = signal_generator.generate_signals(sample_ohlcv_df, "XAU/USD")

        assert isinstance(analysis, SignalAnalysis)
        assert analysis.symbol == "XAU/USD"
        assert isinstance(analysis.current_price, float)
        assert isinstance(analysis.indicators, TechnicalIndicators)
        assert analysis.market_bias in ["BULLISH", "BEARISH", "NEUTRAL"]

    def test_generate_signals_with_bullish_data(self, signal_generator, bullish_ohlcv_df):
        """Test signal generation with bullish data."""
        analysis = signal_generator.generate_signals(bullish_ohlcv_df, "XAU/USD")

        # With bullish data, we expect bullish bias
        assert analysis.market_bias in ["BULLISH", "NEUTRAL"]

    def test_generate_signals_with_bearish_data(self, signal_generator, bearish_ohlcv_df):
        """Test signal generation with bearish data."""
        analysis = signal_generator.generate_signals(bearish_ohlcv_df, "XAU/USD")

        # With bearish data, we expect bearish bias
        assert analysis.market_bias in ["BEARISH", "NEUTRAL"]

    def test_get_signal_summary(self, signal_generator, sample_ohlcv_df):
        """Test getting signal summary."""
        summary = signal_generator.get_signal_summary(sample_ohlcv_df, "XAU/USD")

        assert "has_signal" in summary
        assert "market_bias" in summary
        assert "overall_confidence" in summary
        assert "indicators" in summary
        assert "signal_count" in summary

    def test_create_signal_with_buy(self, signal_generator):
        """Test creating a BUY signal."""
        signal = signal_generator._create_signal(
            direction=SignalDirection.BUY,
            signal_type=SignalType.FIBONACCI_BOUNCE,
            current_price=2045.50,
            atr=5.0,
            reasoning="Test buy signal",
            confidence=0.75,
            symbol="XAU/USD"
        )

        assert signal.direction == SignalDirection.BUY
        assert signal.entry_price == 2045.50
        assert signal.stop_loss < signal.entry_price
        assert signal.take_profit_1 > signal.entry_price
        assert signal.take_profit_2 > signal.take_profit_1
        assert signal.confidence == 0.75
        assert signal.strength == SignalStrength.VERY_STRONG  # 0.75 is >= threshold

    def test_create_signal_with_sell(self, signal_generator):
        """Test creating a SELL signal."""
        signal = signal_generator._create_signal(
            direction=SignalDirection.SELL,
            signal_type=SignalType.RSI_DIVERGENCE,
            current_price=2045.50,
            atr=5.0,
            reasoning="Test sell signal",
            confidence=0.65,
            symbol="XAU/USD"
        )

        assert signal.direction == SignalDirection.SELL
        assert signal.entry_price == 2045.50
        assert signal.stop_loss > signal.entry_price
        assert signal.take_profit_1 < signal.entry_price
        assert signal.take_profit_2 < signal.take_profit_1

    def test_signal_risk_reward_calculation(self, signal_generator):
        """Test risk:reward calculation in signals."""
        signal = signal_generator._create_signal(
            direction=SignalDirection.BUY,
            signal_type=SignalType.BREAKOUT,
            current_price=2000.0,
            atr=10.0,
            reasoning="Test signal",
            confidence=0.70,
            symbol="XAU/USD"
        )

        # Risk reward should be positive
        assert signal.risk_reward_1 > 0
        assert signal.risk_reward_2 > 0
        # TP2 should have higher R:R than TP1
        assert signal.risk_reward_2 > signal.risk_reward_1

    def test_position_size_limits(self, signal_generator):
        """Test that position size is within limits."""
        signal = signal_generator._create_signal(
            direction=SignalDirection.BUY,
            signal_type=SignalType.MACD_CROSSOVER,
            current_price=2000.0,
            atr=5.0,
            reasoning="Test signal",
            confidence=0.60,
            symbol="XAU/USD"
        )

        # Position size should be between 0.01 and 1.0
        assert 0.01 <= signal.position_size_lots <= 1.0

    def test_empty_dataframe(self, signal_generator):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        analysis = signal_generator.generate_signals(empty_df, "XAU/USD")

        assert analysis.current_price == 0
        assert len(analysis.active_signals) == 0

    def test_small_dataframe(self, signal_generator):
        """Test handling of DataFrame with insufficient data."""
        small_df = pd.DataFrame({
            'open': [2000, 2001],
            'high': [2002, 2003],
            'low': [1998, 1999],
            'close': [2001, 2002],
            'volume': [1000, 1100]
        })

        analysis = signal_generator.generate_signals(small_df, "XAU/USD")

        # Should handle gracefully
        assert analysis.current_price == 0 or len(analysis.active_signals) == 0

    def test_signal_strength_classification(self, signal_generator):
        """Test signal strength classification based on confidence."""
        # Very strong (>= 0.75)
        signal_vs = signal_generator._create_signal(
            SignalDirection.BUY, SignalType.REVERSAL,
            2000.0, 5.0, "test", 0.80, "XAU/USD"
        )
        assert signal_vs.strength == SignalStrength.VERY_STRONG

        # Strong (>= 0.65)
        signal_s = signal_generator._create_signal(
            SignalDirection.BUY, SignalType.REVERSAL,
            2000.0, 5.0, "test", 0.70, "XAU/USD"
        )
        assert signal_s.strength == SignalStrength.STRONG

        # Moderate (>= 0.55)
        signal_m = signal_generator._create_signal(
            SignalDirection.BUY, SignalType.REVERSAL,
            2000.0, 5.0, "test", 0.60, "XAU/USD"
        )
        assert signal_m.strength == SignalStrength.MODERATE

        # Weak (< 0.55)
        signal_w = signal_generator._create_signal(
            SignalDirection.BUY, SignalType.REVERSAL,
            2000.0, 5.0, "test", 0.50, "XAU/USD"
        )
        assert signal_w.strength == SignalStrength.WEAK

    def test_market_bias_determination(self, signal_generator, sample_ohlcv_df):
        """Test market bias determination."""
        indicators = signal_generator._get_indicators(sample_ohlcv_df)
        bias = signal_generator._determine_market_bias(indicators)

        assert bias in ["BULLISH", "BEARISH", "NEUTRAL"]


class TestTradeSignal:
    """Tests for TradeSignal dataclass."""

    def test_risk_reward_properties(self):
        """Test risk:reward ratio properties."""
        signal = TradeSignal(
            asset="XAU/USD",
            direction=SignalDirection.BUY,
            signal_type=SignalType.FIBONACCI_BOUNCE,
            entry_price=2000.0,
            stop_loss=1990.0,  # 10 pips risk
            take_profit_1=2020.0,  # 20 pips reward
            take_profit_2=2035.0,  # 35 pips reward
            position_size_lots=0.1,
            risk_amount=100.0,
            confidence=0.70,
            strength=SignalStrength.STRONG,
            expiry_minutes=15,
            reasoning="Test signal"
        )

        assert signal.risk_reward_1 == 2.0  # 20/10
        assert signal.risk_reward_2 == 3.5  # 35/10

    def test_risk_reward_with_zero_risk(self):
        """Test risk:reward when risk is zero."""
        signal = TradeSignal(
            asset="XAU/USD",
            direction=SignalDirection.BUY,
            signal_type=SignalType.BREAKOUT,
            entry_price=2000.0,
            stop_loss=2000.0,  # No risk
            take_profit_1=2010.0,
            take_profit_2=2020.0,
            position_size_lots=0.1,
            risk_amount=0,
            confidence=0.60,
            strength=SignalStrength.MODERATE,
            expiry_minutes=15,
            reasoning="Test signal"
        )

        assert signal.risk_reward_1 == 0
        assert signal.risk_reward_2 == 0
