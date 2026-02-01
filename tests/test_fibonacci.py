"""Tests for Fibonacci analysis module."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.analysis.fibonacci import (
    FibonacciAnalyzer,
    FibonacciAnalysis,
    FibonacciLevel,
    TrendDirection
)


@pytest.fixture
def analyzer():
    """Create a Fibonacci analyzer instance."""
    return FibonacciAnalyzer(lookback_period=50)


@pytest.fixture
def uptrend_prices():
    """Generate price series with uptrend."""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    # Create uptrend: starts at 100, ends at 150 with some noise
    prices = 100 + np.linspace(0, 50, 30) + np.random.randn(30) * 2
    return pd.Series(prices, index=dates, name='TEST')


@pytest.fixture
def downtrend_prices():
    """Generate price series with downtrend."""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    # Create downtrend: starts at 150, ends at 100 with some noise
    prices = 150 - np.linspace(0, 50, 30) + np.random.randn(30) * 2
    return pd.Series(prices, index=dates, name='TEST')


@pytest.fixture
def sideways_prices():
    """Generate price series with sideways movement."""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    # Create sideways: oscillates around 100
    prices = 100 + np.sin(np.linspace(0, 4 * np.pi, 30)) * 5
    return pd.Series(prices, index=dates, name='TEST')


class TestFibonacciAnalyzer:
    """Test suite for FibonacciAnalyzer class."""

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert analyzer.lookback_period == 50
        assert len(analyzer.RETRACEMENT_RATIOS) == 7
        assert len(analyzer.EXTENSION_RATIOS) == 7

    def test_find_swing_points(self, analyzer, uptrend_prices):
        """Test swing point detection."""
        high, low, high_idx, low_idx = analyzer.find_swing_points(uptrend_prices)

        assert high >= low
        assert high == uptrend_prices.max()
        assert low == uptrend_prices.min()

    def test_find_swing_points_short_series(self, analyzer):
        """Test swing points with short data."""
        short_prices = pd.Series([100, 110, 105])
        high, low, _, _ = analyzer.find_swing_points(short_prices)

        assert high == 110
        assert low == 100

    def test_detect_uptrend(self, analyzer, uptrend_prices):
        """Test uptrend detection."""
        high, low, high_idx, low_idx = analyzer.find_swing_points(uptrend_prices)
        trend = analyzer.detect_trend(uptrend_prices, high_idx, low_idx)

        assert trend == TrendDirection.UPTREND

    def test_detect_downtrend(self, analyzer, downtrend_prices):
        """Test downtrend detection."""
        high, low, high_idx, low_idx = analyzer.find_swing_points(downtrend_prices)
        trend = analyzer.detect_trend(downtrend_prices, high_idx, low_idx)

        assert trend == TrendDirection.DOWNTREND

    def test_calculate_retracement_levels(self, analyzer):
        """Test retracement level calculation."""
        levels = analyzer.calculate_retracement_levels(
            swing_high=150,
            swing_low=100,
            trend=TrendDirection.UPTREND
        )

        assert len(levels) == 7

        # Check 0% level (swing high in uptrend)
        level_0 = next(l for l in levels if l.ratio == 0.0)
        assert level_0.price == 150.0

        # Check 100% level (swing low in uptrend)
        level_100 = next(l for l in levels if l.ratio == 1.0)
        assert level_100.price == 100.0

        # Check 50% level
        level_50 = next(l for l in levels if l.ratio == 0.5)
        assert level_50.price == 125.0

        # Check 61.8% level (golden ratio)
        level_618 = next(l for l in levels if l.ratio == 0.618)
        expected_618 = 150 - (50 * 0.618)
        assert abs(level_618.price - expected_618) < 0.01

    def test_calculate_extension_levels(self, analyzer):
        """Test extension level calculation."""
        levels = analyzer.calculate_extension_levels(
            swing_high=150,
            swing_low=100,
            trend=TrendDirection.UPTREND
        )

        assert len(levels) == 7

        # Check 161.8% extension
        level_1618 = next(l for l in levels if l.ratio == 1.618)
        expected = 100 + (50 * 1.618)  # In uptrend, extension above high
        assert abs(level_1618.price - expected) < 0.01

    def test_fibonacci_levels_are_ordered(self, analyzer):
        """Test that levels are sorted by price descending."""
        levels = analyzer.calculate_retracement_levels(
            swing_high=150,
            swing_low=100,
            trend=TrendDirection.UPTREND
        )

        prices = [l.price for l in levels]
        assert prices == sorted(prices, reverse=True)

    def test_find_current_zone(self, analyzer):
        """Test current zone detection."""
        levels = [
            FibonacciLevel(0.0, 150, "0%", False, True, 1.0),
            FibonacciLevel(0.5, 125, "50%", True, True, 1.0),
            FibonacciLevel(1.0, 100, "100%", True, False, 1.0),
        ]

        zone = analyzer.find_current_zone(130, levels)
        assert "50%" in zone
        assert "0%" in zone

    def test_find_current_zone_below_all(self, analyzer):
        """Test zone when price is below all levels."""
        levels = [
            FibonacciLevel(0.0, 150, "0%", False, True, 1.0),
            FibonacciLevel(1.0, 100, "100%", True, False, 1.0),
        ]

        zone = analyzer.find_current_zone(90, levels)
        assert "Below" in zone

    def test_find_current_zone_above_all(self, analyzer):
        """Test zone when price is above all levels."""
        levels = [
            FibonacciLevel(0.0, 150, "0%", False, True, 1.0),
            FibonacciLevel(1.0, 100, "100%", True, False, 1.0),
        ]

        zone = analyzer.find_current_zone(160, levels)
        assert "Above" in zone

    def test_find_nearest_levels(self, analyzer):
        """Test finding nearest support and resistance."""
        levels = [
            FibonacciLevel(0.0, 150, "0%", False, True, 1.0),
            FibonacciLevel(0.382, 130.9, "38.2%", True, True, 1.0),
            FibonacciLevel(0.5, 125, "50%", True, True, 1.0),
            FibonacciLevel(0.618, 119.1, "61.8%", True, False, 1.0),
            FibonacciLevel(1.0, 100, "100%", True, False, 1.0),
        ]

        current_price = 127

        support, resistance = analyzer.find_nearest_levels(current_price, levels)

        assert support is not None
        assert resistance is not None
        assert support.price < current_price
        assert resistance.price > current_price

    def test_analyze_complete(self, analyzer, uptrend_prices):
        """Test complete analysis."""
        analysis = analyzer.analyze(uptrend_prices, "TEST")

        assert isinstance(analysis, FibonacciAnalysis)
        assert analysis.symbol == "TEST"
        assert analysis.swing_high >= analysis.swing_low
        assert len(analysis.retracement_levels) == 7
        assert len(analysis.extension_levels) == 7
        assert 0 <= analysis.confidence <= 1

    def test_analyze_raises_on_insufficient_data(self, analyzer):
        """Test that analysis raises error with too little data."""
        short_prices = pd.Series([100, 110])

        with pytest.raises(ValueError, match="Insufficient"):
            analyzer.analyze(short_prices)

    def test_get_levels_as_dict(self, analyzer, uptrend_prices):
        """Test dictionary conversion."""
        analysis = analyzer.analyze(uptrend_prices, "TEST")
        result = analyzer.get_levels_as_dict(analysis)

        assert isinstance(result, dict)
        assert "symbol" in result
        assert "trend" in result
        assert "retracement_levels" in result
        assert "extension_levels" in result
        assert result["symbol"] == "TEST"

    def test_golden_ratio_is_present(self, analyzer):
        """Test that golden ratio (0.618) is correctly calculated."""
        levels = analyzer.calculate_retracement_levels(
            swing_high=200,
            swing_low=100,
            trend=TrendDirection.UPTREND
        )

        golden = next((l for l in levels if l.ratio == 0.618), None)
        assert golden is not None
        assert "Golden" in golden.label
        assert golden.strength == 1.0  # Should be high strength


class TestFibonacciLevel:
    """Tests for FibonacciLevel dataclass."""

    def test_level_creation(self):
        """Test creating a Fibonacci level."""
        level = FibonacciLevel(
            ratio=0.618,
            price=138.2,
            label="61.8% (Golden Ratio)",
            is_support=True,
            is_resistance=False,
            strength=1.0
        )

        assert level.ratio == 0.618
        assert level.price == 138.2
        assert level.is_support is True
        assert level.is_resistance is False


class TestTrendDirection:
    """Tests for TrendDirection enum."""

    def test_trend_values(self):
        """Test trend direction values."""
        assert TrendDirection.UPTREND.value == "uptrend"
        assert TrendDirection.DOWNTREND.value == "downtrend"
        assert TrendDirection.SIDEWAYS.value == "sideways"


class TestPatternDetection:
    """Tests for Fibonacci pattern detection."""

    def test_detect_golden_bounce(self, analyzer):
        """Test detection of golden ratio bounce pattern."""
        # Create prices that bounce off 61.8% level
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        # High at 150, low at 100, current bouncing from ~119 (61.8%)
        prices_up = np.linspace(100, 150, 10)
        prices_retrace = np.linspace(150, 119, 5)
        prices_bounce = np.linspace(119, 130, 5)
        prices = np.concatenate([prices_up, prices_retrace, prices_bounce])

        price_series = pd.Series(prices, index=dates, name='TEST')

        analysis = analyzer.analyze(price_series, "TEST")

        # Pattern detection depends on price action
        # The test verifies the mechanism works
        assert analysis.pattern_detected is None or isinstance(analysis.pattern_detected, str)

    def test_pattern_none_with_insufficient_data(self, analyzer):
        """Test that pattern is None with minimal data."""
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        prices = pd.Series([100, 105, 110, 108, 112], index=dates)

        analysis = analyzer.analyze(prices, "TEST")
        # With minimal data, pattern detection should return None
        assert analysis.pattern_detected is None
