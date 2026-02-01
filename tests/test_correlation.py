"""Tests for correlation analysis module."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from app.analysis.correlation import (
    CorrelationAnalyzer,
    CorrelationMatrix,
    CorrelationPair,
    CorrelationStrength,
    SentimentCorrelation
)


@pytest.fixture
def analyzer():
    """Create a correlation analyzer instance."""
    return CorrelationAnalyzer()


@pytest.fixture
def sample_price_data():
    """Generate sample price data for multiple assets."""
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')

    # Create correlated assets
    base = 100 + np.cumsum(np.random.randn(50))

    return {
        'AAPL': pd.Series(base + np.random.randn(50) * 2, index=dates, name='AAPL'),
        'MSFT': pd.Series(base * 1.1 + np.random.randn(50) * 2, index=dates, name='MSFT'),  # Positively correlated
        'GOLD': pd.Series(200 - base * 0.5 + np.random.randn(50) * 2, index=dates, name='GOLD'),  # Negatively correlated
        'RANDOM': pd.Series(100 + np.random.randn(50) * 10, index=dates, name='RANDOM'),  # Uncorrelated
    }


@pytest.fixture
def perfectly_correlated_data():
    """Generate perfectly correlated price data."""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    base = np.cumsum(np.random.randn(30))

    return {
        'A': pd.Series(base, index=dates, name='A'),
        'B': pd.Series(base * 2, index=dates, name='B'),  # Perfect positive correlation
    }


@pytest.fixture
def inversely_correlated_data():
    """Generate inversely correlated price data."""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    base = np.cumsum(np.random.randn(30))

    return {
        'A': pd.Series(base, index=dates, name='A'),
        'B': pd.Series(-base, index=dates, name='B'),  # Perfect negative correlation
    }


class TestCorrelationStrength:
    """Tests for correlation strength classification."""

    def test_strong_positive(self, analyzer):
        """Test strong positive classification."""
        assert analyzer.classify_strength(0.8) == CorrelationStrength.STRONG_POSITIVE
        assert analyzer.classify_strength(0.95) == CorrelationStrength.STRONG_POSITIVE

    def test_moderate_positive(self, analyzer):
        """Test moderate positive classification."""
        assert analyzer.classify_strength(0.5) == CorrelationStrength.MODERATE_POSITIVE
        assert analyzer.classify_strength(0.69) == CorrelationStrength.MODERATE_POSITIVE

    def test_weak_positive(self, analyzer):
        """Test weak positive classification."""
        assert analyzer.classify_strength(0.1) == CorrelationStrength.WEAK_POSITIVE
        assert analyzer.classify_strength(0.29) == CorrelationStrength.WEAK_POSITIVE

    def test_weak_negative(self, analyzer):
        """Test weak negative classification."""
        assert analyzer.classify_strength(-0.1) == CorrelationStrength.WEAK_NEGATIVE
        assert analyzer.classify_strength(-0.29) == CorrelationStrength.WEAK_NEGATIVE

    def test_moderate_negative(self, analyzer):
        """Test moderate negative classification."""
        assert analyzer.classify_strength(-0.5) == CorrelationStrength.MODERATE_NEGATIVE
        assert analyzer.classify_strength(-0.69) == CorrelationStrength.MODERATE_NEGATIVE

    def test_strong_negative(self, analyzer):
        """Test strong negative classification."""
        assert analyzer.classify_strength(-0.8) == CorrelationStrength.STRONG_NEGATIVE
        assert analyzer.classify_strength(-0.95) == CorrelationStrength.STRONG_NEGATIVE


class TestCorrelationCalculation:
    """Tests for correlation calculation methods."""

    def test_calculate_returns_log(self, analyzer):
        """Test log returns calculation."""
        prices = pd.Series([100, 110, 105, 115])
        returns = analyzer.calculate_returns(prices, method="log")

        assert len(returns) == 3
        assert not np.isnan(returns).any()

    def test_calculate_returns_simple(self, analyzer):
        """Test simple returns calculation."""
        prices = pd.Series([100, 110, 105, 115])
        returns = analyzer.calculate_returns(prices, method="simple")

        assert len(returns) == 3
        # First return should be 10%
        assert abs(returns.iloc[0] - 0.1) < 0.001

    def test_calculate_correlation_pearson(self, analyzer):
        """Test Pearson correlation."""
        s1 = pd.Series([1, 2, 3, 4, 5])
        s2 = pd.Series([2, 4, 6, 8, 10])

        corr, n = analyzer.calculate_correlation(s1, s2, method="pearson")

        assert abs(corr - 1.0) < 0.001  # Perfect correlation
        assert n == 5

    def test_calculate_correlation_insufficient_data(self, analyzer):
        """Test correlation with insufficient data."""
        s1 = pd.Series([1, 2])
        s2 = pd.Series([2, 4])

        corr, n = analyzer.calculate_correlation(s1, s2)

        assert corr == 0.0
        assert n == 0


class TestCorrelationMatrix:
    """Tests for correlation matrix building."""

    def test_build_matrix_basic(self, analyzer, sample_price_data):
        """Test building basic correlation matrix."""
        matrix = analyzer.build_correlation_matrix(sample_price_data, period="30d")

        assert isinstance(matrix, CorrelationMatrix)
        assert len(matrix.assets) == 4
        assert matrix.period == "30d"
        assert matrix.matrix.shape == (4, 4)

    def test_matrix_diagonal_is_one(self, analyzer, sample_price_data):
        """Test that diagonal elements are 1 (self-correlation)."""
        matrix = analyzer.build_correlation_matrix(sample_price_data)

        for asset in matrix.assets:
            assert abs(matrix.matrix.loc[asset, asset] - 1.0) < 0.001

    def test_matrix_is_symmetric(self, analyzer, sample_price_data):
        """Test that correlation matrix is symmetric."""
        matrix = analyzer.build_correlation_matrix(sample_price_data)

        for i, a1 in enumerate(matrix.assets):
            for j, a2 in enumerate(matrix.assets):
                assert abs(matrix.matrix.loc[a1, a2] - matrix.matrix.loc[a2, a1]) < 0.001

    def test_matrix_finds_strongest_positive(self, analyzer, perfectly_correlated_data):
        """Test finding strongest positive correlation."""
        matrix = analyzer.build_correlation_matrix(perfectly_correlated_data)

        assert matrix.strongest_positive is not None
        assert abs(matrix.strongest_positive.correlation - 1.0) < 0.001

    def test_matrix_finds_strongest_negative(self, analyzer, inversely_correlated_data):
        """Test finding strongest negative correlation."""
        matrix = analyzer.build_correlation_matrix(inversely_correlated_data)

        assert matrix.strongest_negative is not None
        assert abs(matrix.strongest_negative.correlation - (-1.0)) < 0.001

    def test_matrix_requires_two_assets(self, analyzer):
        """Test that matrix requires at least 2 assets."""
        single_asset = {'AAPL': pd.Series([100, 110, 105])}

        with pytest.raises(ValueError, match="at least 2"):
            analyzer.build_correlation_matrix(single_asset)


class TestRollingCorrelation:
    """Tests for rolling correlation calculation."""

    def test_rolling_correlation_basic(self, analyzer):
        """Test basic rolling correlation."""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        s1 = pd.Series(np.cumsum(np.random.randn(30)), index=dates)
        s2 = pd.Series(np.cumsum(np.random.randn(30)), index=dates)

        rolling = analyzer.calculate_rolling_correlation(s1, s2, window=10)

        assert len(rolling) == 30
        # First window-1 values should be NaN
        assert rolling.isna().sum() >= 9

    def test_rolling_correlation_insufficient_data(self, analyzer):
        """Test rolling correlation with insufficient data."""
        s1 = pd.Series([1, 2, 3])
        s2 = pd.Series([2, 4, 6])

        rolling = analyzer.calculate_rolling_correlation(s1, s2, window=10)

        assert len(rolling) == 0


class TestSentimentCorrelation:
    """Tests for sentiment-price correlation."""

    def test_sentiment_correlation_basic(self, analyzer):
        """Test basic sentiment correlation calculation."""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        prices = pd.Series(100 + np.cumsum(np.random.randn(30)), index=dates, name='TEST')
        sentiment = pd.Series(np.random.randn(30), index=dates)

        result = analyzer.calculate_sentiment_correlation(prices, sentiment)

        assert isinstance(result, SentimentCorrelation)
        assert -1 <= result.correlation <= 1
        assert 0 <= result.lag_days <= 5

    def test_sentiment_correlation_with_lag(self, analyzer):
        """Test sentiment correlation finds optimal lag."""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')

        # Create sentiment that leads price by 2 days
        base_sentiment = np.random.randn(30)
        prices = pd.Series(100 + np.cumsum(np.roll(base_sentiment, 2)), index=dates, name='TEST')
        sentiment = pd.Series(base_sentiment, index=dates)

        result = analyzer.calculate_sentiment_correlation(prices, sentiment, max_lag=5)

        # The analyzer should detect some correlation
        assert isinstance(result.lag_days, int)
        assert result.lag_days >= 0


class TestCorrelationPairs:
    """Tests for finding correlated pairs."""

    def test_find_correlated_pairs(self, analyzer, sample_price_data):
        """Test finding pairs above threshold."""
        matrix = analyzer.build_correlation_matrix(sample_price_data)
        pairs = analyzer.find_correlated_pairs(matrix, threshold=0.3)

        assert isinstance(pairs, list)
        for pair in pairs:
            assert isinstance(pair, CorrelationPair)
            assert abs(pair.correlation) >= 0.3

    def test_pairs_sorted_by_strength(self, analyzer, sample_price_data):
        """Test that pairs are sorted by absolute correlation."""
        matrix = analyzer.build_correlation_matrix(sample_price_data)
        pairs = analyzer.find_correlated_pairs(matrix, threshold=0.0)

        for i in range(len(pairs) - 1):
            assert abs(pairs[i].correlation) >= abs(pairs[i + 1].correlation)


class TestCorrelationChanges:
    """Tests for correlation regime change detection."""

    def test_detect_changes_basic(self, analyzer):
        """Test basic regime change detection."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        s1 = pd.Series(np.cumsum(np.random.randn(50)), index=dates)
        s2 = pd.Series(np.cumsum(np.random.randn(50)), index=dates)

        result = analyzer.detect_correlation_changes(s1, s2)

        assert isinstance(result, dict)
        assert "regime_change" in result
        assert "short_term_corr" in result
        assert "long_term_corr" in result
        assert "divergence" in result
        assert "interpretation" in result

    def test_detect_changes_insufficient_data(self, analyzer):
        """Test regime change with insufficient data."""
        s1 = pd.Series([1, 2, 3])
        s2 = pd.Series([2, 4, 6])

        result = analyzer.detect_correlation_changes(s1, s2)

        assert result["regime_change"] is False


class TestMatrixSerialization:
    """Tests for matrix serialization."""

    def test_matrix_to_dict(self, analyzer, sample_price_data):
        """Test converting matrix to dictionary."""
        matrix = analyzer.build_correlation_matrix(sample_price_data)
        result = analyzer.matrix_to_dict(matrix)

        assert isinstance(result, dict)
        assert "assets" in result
        assert "matrix" in result
        assert "period" in result
        assert "timestamp" in result
        assert "strongest_positive" in result
        assert "strongest_negative" in result

    def test_matrix_dict_is_json_serializable(self, analyzer, sample_price_data):
        """Test that matrix dict can be serialized to JSON."""
        import json

        matrix = analyzer.build_correlation_matrix(sample_price_data)
        result = analyzer.matrix_to_dict(matrix)

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
