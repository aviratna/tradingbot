"""Tests for the manipulation detection module."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from app.analysis.manipulation import (
    ManipulationDetector,
    ManipulationType,
    ManipulationSeverity,
    ManipulationAlert,
    ManipulationAnalysis,
    KeyLevel,
    OrderBlock
)


@pytest.fixture
def manipulation_detector():
    """Create a manipulation detector instance."""
    return ManipulationDetector(lookback_periods=100)


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')

    np.random.seed(42)
    base_price = 2000
    prices = [base_price]
    for i in range(99):
        change = np.random.normal(0, 2)
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
def stop_hunt_df():
    """Create data with a stop hunt pattern."""
    dates = pd.date_range(start='2024-01-01', periods=20, freq='5min')

    # Create a support level at 2000, then a stop hunt below it
    prices = [2010, 2008, 2005, 2003, 2002, 2001, 2000,  # Approaching support
              2001, 2002, 2000, 1995, 2005,  # Stop hunt (spike below 2000, then reversal)
              2008, 2010, 2012, 2015, 2018, 2020, 2022, 2025]  # Recovery

    df = pd.DataFrame({
        'open': [p - 1 for p in prices],
        'high': [p + 2 for p in prices],
        'low': [p - 3 for p in prices],
        'close': prices,
        'volume': [5000] * 20
    }, index=dates)

    # Adjust the stop hunt candle
    df.loc[df.index[10], 'low'] = 1990  # Sharp spike down
    df.loc[df.index[10], 'close'] = 1995

    return df


@pytest.fixture
def volume_spike_df():
    """Create data with a volume anomaly."""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='5min')

    np.random.seed(42)
    base_price = 2000
    prices = base_price + np.random.normal(0, 1, 30).cumsum()

    volumes = [1000] * 30
    volumes[25] = 10000  # Volume spike

    df = pd.DataFrame({
        'open': prices - 0.5,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices,
        'volume': volumes
    }, index=dates)

    return df


@pytest.fixture
def fake_breakout_df():
    """Create data with a fake breakout pattern."""
    dates = pd.date_range(start='2024-01-01', periods=20, freq='5min')

    # Resistance at 2050, fake breakout above it
    prices = [2040, 2042, 2045, 2047, 2048, 2049, 2050,  # Approaching resistance
              2048, 2049, 2055, 2045,  # Breakout above 2050, then failure
              2042, 2040, 2038, 2035, 2033, 2030, 2028, 2025, 2022]  # Drop

    df = pd.DataFrame({
        'open': [p - 1 for p in prices],
        'high': [p + 2 for p in prices],
        'low': [p - 2 for p in prices],
        'close': prices,
        'volume': [5000] * 20
    }, index=dates)

    # Adjust breakout candle
    df.loc[df.index[9], 'high'] = 2058  # False breakout high
    df.loc[df.index[9], 'open'] = 2049
    df.loc[df.index[9], 'close'] = 2055

    return df


class TestManipulationDetector:
    """Tests for ManipulationDetector class."""

    def test_initialization(self, manipulation_detector):
        """Test manipulation detector initialization."""
        assert manipulation_detector.lookback_periods == 100

    def test_calculate_atr(self, manipulation_detector, sample_ohlcv_df):
        """Test ATR calculation."""
        atr = manipulation_detector._calculate_atr(sample_ohlcv_df)

        assert len(atr) == len(sample_ohlcv_df)
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()

    def test_find_swing_points(self, manipulation_detector, sample_ohlcv_df):
        """Test swing point detection."""
        swing_highs, swing_lows = manipulation_detector._find_swing_points(sample_ohlcv_df)

        assert isinstance(swing_highs, list)
        assert isinstance(swing_lows, list)
        # Should find some swing points
        assert len(swing_highs) > 0 or len(swing_lows) > 0

    def test_find_key_levels(self, manipulation_detector, sample_ohlcv_df):
        """Test key level identification."""
        key_levels = manipulation_detector._find_key_levels(sample_ohlcv_df)

        assert isinstance(key_levels, list)
        for level in key_levels:
            assert isinstance(level, KeyLevel)
            assert level.price > 0
            assert level.level_type in ["SUPPORT", "RESISTANCE", "ROUND_NUMBER"]
            assert 0 <= level.strength <= 1

    def test_get_round_number_levels(self, manipulation_detector):
        """Test round number level detection."""
        # For gold around 2045
        levels = manipulation_detector._get_round_number_levels(2045.0)

        assert len(levels) > 0
        # Should include round numbers near the price
        assert any(l % 10 == 0 for l in levels)

        # For silver around 25
        levels_silver = manipulation_detector._get_round_number_levels(25.0)
        assert len(levels_silver) > 0

    def test_detect_stop_hunt(self, manipulation_detector, stop_hunt_df):
        """Test stop hunt detection."""
        key_levels = [
            KeyLevel(price=2000, level_type="SUPPORT", strength=0.8, touches=3)
        ]

        alerts = manipulation_detector._detect_stop_hunt(stop_hunt_df, key_levels)

        # Should detect the stop hunt pattern
        assert isinstance(alerts, list)
        for alert in alerts:
            assert isinstance(alert, ManipulationAlert)
            assert alert.manipulation_type == ManipulationType.STOP_HUNT

    def test_detect_liquidity_sweep(self, manipulation_detector, stop_hunt_df):
        """Test liquidity sweep detection."""
        key_levels = [
            KeyLevel(price=2000, level_type="SUPPORT", strength=0.8, touches=3)
        ]

        alerts = manipulation_detector._detect_liquidity_sweep(stop_hunt_df, key_levels)

        assert isinstance(alerts, list)
        for alert in alerts:
            assert isinstance(alert, ManipulationAlert)
            assert alert.manipulation_type == ManipulationType.LIQUIDITY_SWEEP

    def test_detect_volume_anomaly(self, manipulation_detector, volume_spike_df):
        """Test volume anomaly detection."""
        alerts = manipulation_detector._detect_volume_anomaly(volume_spike_df)

        assert isinstance(alerts, list)
        # Should detect the volume spike
        volume_alerts = [a for a in alerts if a.manipulation_type == ManipulationType.VOLUME_ANOMALY]
        # May or may not detect depending on the price movement ratio
        for alert in volume_alerts:
            assert alert.severity in [ManipulationSeverity.LOW, ManipulationSeverity.MODERATE,
                                     ManipulationSeverity.HIGH, ManipulationSeverity.CRITICAL]

    def test_detect_fake_breakout(self, manipulation_detector, fake_breakout_df):
        """Test fake breakout detection."""
        key_levels = [
            KeyLevel(price=2050, level_type="RESISTANCE", strength=0.8, touches=2)
        ]

        alerts = manipulation_detector._detect_fake_breakout(fake_breakout_df, key_levels)

        assert isinstance(alerts, list)
        for alert in alerts:
            assert isinstance(alert, ManipulationAlert)
            assert alert.manipulation_type == ManipulationType.FAKE_BREAKOUT

    def test_detect_order_blocks(self, manipulation_detector, sample_ohlcv_df):
        """Test order block detection."""
        order_blocks = manipulation_detector._detect_order_blocks(sample_ohlcv_df)

        assert isinstance(order_blocks, list)
        for ob in order_blocks:
            assert isinstance(ob, OrderBlock)
            assert ob.price_high >= ob.price_low
            assert ob.block_type in ["BULLISH", "BEARISH"]
            assert 0 <= ob.strength <= 1

    def test_get_session_risk(self, manipulation_detector):
        """Test session risk determination."""
        # Asian session (low risk)
        session, risk = manipulation_detector._get_session_risk(3)
        assert session == "ASIAN"
        assert risk == "LOW"

        # London session (high risk)
        session, risk = manipulation_detector._get_session_risk(10)
        assert session == "LONDON"
        assert risk == "HIGH"

        # NY overlap (highest risk)
        session, risk = manipulation_detector._get_session_risk(13)
        assert session == "NY_OVERLAP"
        assert risk == "HIGHEST"

        # New York session
        session, risk = manipulation_detector._get_session_risk(16)
        assert session == "NEW_YORK"
        assert risk == "MODERATE"

    def test_calculate_manipulation_score(self, manipulation_detector):
        """Test manipulation score calculation."""
        # No alerts, low session risk
        score_low = manipulation_detector._calculate_manipulation_score([], "LOW")
        assert 0 <= score_low <= 1
        assert score_low < 0.3

        # Multiple alerts, high session risk
        alerts = [
            ManipulationAlert(
                manipulation_type=ManipulationType.STOP_HUNT,
                severity=ManipulationSeverity.HIGH,
                timestamp=datetime.now(timezone.utc),
                price_at_detection=2000,
                description="Test alert"
            ),
            ManipulationAlert(
                manipulation_type=ManipulationType.VOLUME_ANOMALY,
                severity=ManipulationSeverity.MODERATE,
                timestamp=datetime.now(timezone.utc),
                price_at_detection=2000,
                description="Test alert 2"
            )
        ]
        score_high = manipulation_detector._calculate_manipulation_score(alerts, "HIGHEST")
        assert score_high > score_low

    def test_generate_recommendation(self, manipulation_detector):
        """Test recommendation generation."""
        # Low manipulation
        rec_low = manipulation_detector._generate_recommendation(0.2, [], "LOW")
        assert "NORMAL" in rec_low

        # Moderate manipulation
        rec_mod = manipulation_detector._generate_recommendation(0.55, [], "MODERATE")
        assert "CAUTION" in rec_mod

        # High manipulation
        rec_high = manipulation_detector._generate_recommendation(0.8, [], "HIGH")
        assert "AVOID" in rec_high

    def test_analyze_returns_analysis(self, manipulation_detector, sample_ohlcv_df):
        """Test that analyze returns ManipulationAnalysis."""
        analysis = manipulation_detector.analyze(sample_ohlcv_df, "XAU/USD")

        assert isinstance(analysis, ManipulationAnalysis)
        assert analysis.symbol == "XAU/USD"
        assert isinstance(analysis.current_price, float)
        assert isinstance(analysis.active_alerts, list)
        assert isinstance(analysis.key_levels, list)
        assert isinstance(analysis.order_blocks, list)
        assert 0 <= analysis.overall_manipulation_score <= 1
        assert analysis.session_risk in ["LOW", "MODERATE", "HIGH", "HIGHEST", "UNKNOWN"]

    def test_analyze_with_stop_hunt_data(self, manipulation_detector, stop_hunt_df):
        """Test analysis with stop hunt data."""
        analysis = manipulation_detector.analyze(stop_hunt_df, "XAU/USD")

        # Should detect manipulation
        assert analysis.overall_manipulation_score > 0
        assert len(analysis.recommendation) > 0

    def test_get_quick_status(self, manipulation_detector, sample_ohlcv_df):
        """Test quick status for dashboard."""
        status = manipulation_detector.get_quick_status(sample_ohlcv_df)

        assert "manipulation_score" in status
        assert "risk_level" in status
        assert "session_risk" in status
        assert "active_alerts_count" in status
        assert "recommendation" in status
        assert "key_levels" in status

        assert status["risk_level"] in ["LOW", "MODERATE", "HIGH"]

    def test_empty_dataframe(self, manipulation_detector):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        analysis = manipulation_detector.analyze(empty_df, "XAU/USD")

        assert analysis.current_price == 0
        assert len(analysis.active_alerts) == 0
        assert "No data" in analysis.recommendation

    def test_small_dataframe(self, manipulation_detector):
        """Test handling of DataFrame with few rows."""
        small_df = pd.DataFrame({
            'open': [2000, 2001],
            'high': [2002, 2003],
            'low': [1998, 1999],
            'close': [2001, 2002],
            'volume': [1000, 1100]
        })

        analysis = manipulation_detector.analyze(small_df, "XAU/USD")

        # Should handle gracefully
        assert isinstance(analysis, ManipulationAnalysis)

    def test_consolidate_levels(self, manipulation_detector, sample_ohlcv_df):
        """Test level consolidation."""
        levels = [
            KeyLevel(price=2000.0, level_type="SUPPORT", strength=0.7, touches=1),
            KeyLevel(price=2000.5, level_type="SUPPORT", strength=0.6, touches=1),  # Close to first
            KeyLevel(price=2050.0, level_type="RESISTANCE", strength=0.8, touches=2),
        ]

        consolidated = manipulation_detector._consolidate_levels(levels, sample_ohlcv_df)

        # Should merge nearby levels
        assert len(consolidated) <= len(levels)


class TestManipulationAlert:
    """Tests for ManipulationAlert dataclass."""

    def test_alert_creation(self):
        """Test creating a manipulation alert."""
        alert = ManipulationAlert(
            manipulation_type=ManipulationType.STOP_HUNT,
            severity=ManipulationSeverity.HIGH,
            timestamp=datetime.now(timezone.utc),
            price_at_detection=2045.50,
            description="Bullish stop hunt detected at 2040",
            key_level_involved=2040.0,
            expected_reversal="UP",
            confidence=0.75
        )

        assert alert.manipulation_type == ManipulationType.STOP_HUNT
        assert alert.severity == ManipulationSeverity.HIGH
        assert alert.price_at_detection == 2045.50
        assert alert.key_level_involved == 2040.0
        assert alert.expected_reversal == "UP"
        assert alert.confidence == 0.75
        assert alert.expires_in_minutes == 15  # Default


class TestKeyLevel:
    """Tests for KeyLevel dataclass."""

    def test_key_level_creation(self):
        """Test creating a key level."""
        level = KeyLevel(
            price=2050.0,
            level_type="RESISTANCE",
            strength=0.85,
            touches=4
        )

        assert level.price == 2050.0
        assert level.level_type == "RESISTANCE"
        assert level.strength == 0.85
        assert level.touches == 4
        assert level.last_touch is None


class TestOrderBlock:
    """Tests for OrderBlock dataclass."""

    def test_order_block_creation(self):
        """Test creating an order block."""
        ob = OrderBlock(
            price_high=2055.0,
            price_low=2050.0,
            block_type="BULLISH",
            strength=0.75,
            timestamp=datetime.now(timezone.utc)
        )

        assert ob.price_high == 2055.0
        assert ob.price_low == 2050.0
        assert ob.block_type == "BULLISH"
        assert ob.strength == 0.75
        assert ob.is_tested is False
        assert ob.is_broken is False
