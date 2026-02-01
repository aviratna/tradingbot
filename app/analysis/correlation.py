"""Correlation analysis for cross-asset and sentiment correlations."""
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd


class CorrelationStrength(Enum):
    """Correlation strength classification."""
    STRONG_POSITIVE = "strong_positive"      # > 0.7
    MODERATE_POSITIVE = "moderate_positive"  # 0.3 to 0.7
    WEAK_POSITIVE = "weak_positive"          # 0 to 0.3
    WEAK_NEGATIVE = "weak_negative"          # -0.3 to 0
    MODERATE_NEGATIVE = "moderate_negative"  # -0.7 to -0.3
    STRONG_NEGATIVE = "strong_negative"      # < -0.7


@dataclass
class CorrelationPair:
    """Correlation between two assets."""
    asset1: str
    asset2: str
    correlation: float
    strength: CorrelationStrength
    period: str  # e.g., "7d", "30d", "90d"
    sample_size: int


@dataclass
class CorrelationMatrix:
    """Complete correlation matrix for multiple assets."""
    assets: List[str]
    matrix: pd.DataFrame
    period: str
    timestamp: str
    strongest_positive: Optional[CorrelationPair]
    strongest_negative: Optional[CorrelationPair]


@dataclass
class SentimentCorrelation:
    """Correlation between sentiment and price movement."""
    asset: str
    sentiment_source: str  # 'news', 'social', 'combined'
    correlation: float
    strength: CorrelationStrength
    lag_days: int  # Lag between sentiment and price (0 = same day)
    p_value: float


class CorrelationAnalyzer:
    """
    Analyzes correlations between assets, sentiment, and market data.

    Provides cross-asset correlation matrices, sentiment-price correlations,
    and identifies significant relationships for trading insights.
    """

    def __init__(self):
        self.min_periods = 5  # Minimum data points for correlation

    @staticmethod
    def classify_strength(correlation: float) -> CorrelationStrength:
        """
        Classify correlation coefficient into strength categories.

        Args:
            correlation: Correlation coefficient (-1 to 1)

        Returns:
            CorrelationStrength enum
        """
        if correlation > 0.7:
            return CorrelationStrength.STRONG_POSITIVE
        elif correlation > 0.3:
            return CorrelationStrength.MODERATE_POSITIVE
        elif correlation > 0:
            return CorrelationStrength.WEAK_POSITIVE
        elif correlation > -0.3:
            return CorrelationStrength.WEAK_NEGATIVE
        elif correlation > -0.7:
            return CorrelationStrength.MODERATE_NEGATIVE
        else:
            return CorrelationStrength.STRONG_NEGATIVE

    def calculate_returns(
        self,
        prices: pd.Series,
        method: str = "log"
    ) -> pd.Series:
        """
        Calculate returns from price series.

        Args:
            prices: Series of prices
            method: 'log' for log returns, 'simple' for simple returns

        Returns:
            Series of returns
        """
        if method == "log":
            return np.log(prices / prices.shift(1)).dropna()
        else:
            return prices.pct_change().dropna()

    def calculate_correlation(
        self,
        series1: pd.Series,
        series2: pd.Series,
        method: str = "pearson"
    ) -> Tuple[float, int]:
        """
        Calculate correlation between two series.

        Args:
            series1: First data series
            series2: Second data series
            method: 'pearson', 'spearman', or 'kendall'

        Returns:
            Tuple of (correlation coefficient, sample size)
        """
        # Align series by index
        aligned = pd.concat([series1, series2], axis=1).dropna()

        if len(aligned) < self.min_periods:
            return 0.0, 0

        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method=method)
        return round(corr, 4), len(aligned)

    def build_correlation_matrix(
        self,
        price_data: Dict[str, pd.Series],
        period: str = "30d",
        use_returns: bool = True
    ) -> CorrelationMatrix:
        """
        Build correlation matrix for multiple assets.

        Args:
            price_data: Dictionary mapping asset names to price series
            period: Period label for the analysis
            use_returns: If True, calculate correlations on returns

        Returns:
            CorrelationMatrix object
        """
        assets = list(price_data.keys())

        if len(assets) < 2:
            raise ValueError("Need at least 2 assets for correlation matrix")

        # Convert to returns if requested
        data = {}
        for asset, prices in price_data.items():
            if use_returns:
                data[asset] = self.calculate_returns(prices)
            else:
                data[asset] = prices

        # Build DataFrame
        df = pd.DataFrame(data)

        # Calculate correlation matrix
        corr_matrix = df.corr()

        # Find strongest correlations (excluding self-correlations)
        strongest_positive = None
        strongest_negative = None
        max_pos = 0.0
        max_neg = 0.0

        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i >= j:  # Skip diagonal and lower triangle
                    continue

                corr = corr_matrix.loc[asset1, asset2]

                if pd.isna(corr):
                    continue

                if corr > max_pos:
                    max_pos = corr
                    strongest_positive = CorrelationPair(
                        asset1=asset1,
                        asset2=asset2,
                        correlation=round(corr, 4),
                        strength=self.classify_strength(corr),
                        period=period,
                        sample_size=len(df.dropna())
                    )

                if corr < max_neg:
                    max_neg = corr
                    strongest_negative = CorrelationPair(
                        asset1=asset1,
                        asset2=asset2,
                        correlation=round(corr, 4),
                        strength=self.classify_strength(corr),
                        period=period,
                        sample_size=len(df.dropna())
                    )

        from datetime import datetime
        return CorrelationMatrix(
            assets=assets,
            matrix=corr_matrix.round(4),
            period=period,
            timestamp=datetime.now().isoformat(),
            strongest_positive=strongest_positive,
            strongest_negative=strongest_negative
        )

    def calculate_rolling_correlation(
        self,
        series1: pd.Series,
        series2: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate rolling correlation between two series.

        Args:
            series1: First data series
            series2: Second data series
            window: Rolling window size

        Returns:
            Series of rolling correlations
        """
        aligned = pd.concat([series1, series2], axis=1).dropna()

        if len(aligned) < window:
            return pd.Series(dtype=float)

        return aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])

    def calculate_sentiment_correlation(
        self,
        prices: pd.Series,
        sentiment_scores: pd.Series,
        max_lag: int = 5
    ) -> SentimentCorrelation:
        """
        Calculate correlation between sentiment and price movement.

        Args:
            prices: Price series
            sentiment_scores: Sentiment score series (same index as prices)
            max_lag: Maximum lag days to test

        Returns:
            SentimentCorrelation with the best lag
        """
        returns = self.calculate_returns(prices)

        best_corr = 0.0
        best_lag = 0

        for lag in range(max_lag + 1):
            if lag > 0:
                lagged_sentiment = sentiment_scores.shift(lag)
            else:
                lagged_sentiment = sentiment_scores

            aligned = pd.concat([returns, lagged_sentiment], axis=1).dropna()

            if len(aligned) < self.min_periods:
                continue

            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        return SentimentCorrelation(
            asset=prices.name or "unknown",
            sentiment_source="combined",
            correlation=round(best_corr, 4),
            strength=self.classify_strength(best_corr),
            lag_days=best_lag,
            p_value=0.0  # Would need scipy for actual p-value
        )

    def find_correlated_pairs(
        self,
        correlation_matrix: CorrelationMatrix,
        threshold: float = 0.5
    ) -> List[CorrelationPair]:
        """
        Find pairs with correlation above threshold.

        Args:
            correlation_matrix: CorrelationMatrix object
            threshold: Minimum absolute correlation

        Returns:
            List of CorrelationPair objects
        """
        pairs = []
        matrix = correlation_matrix.matrix
        assets = correlation_matrix.assets

        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i >= j:
                    continue

                corr = matrix.loc[asset1, asset2]

                if pd.isna(corr):
                    continue

                if abs(corr) >= threshold:
                    pairs.append(CorrelationPair(
                        asset1=asset1,
                        asset2=asset2,
                        correlation=round(corr, 4),
                        strength=self.classify_strength(corr),
                        period=correlation_matrix.period,
                        sample_size=0  # Would need to track this
                    ))

        return sorted(pairs, key=lambda x: abs(x.correlation), reverse=True)

    def detect_correlation_changes(
        self,
        series1: pd.Series,
        series2: pd.Series,
        short_window: int = 10,
        long_window: int = 30
    ) -> Dict[str, Any]:
        """
        Detect significant changes in correlation regime.

        Args:
            series1: First data series
            series2: Second data series
            short_window: Short-term correlation window
            long_window: Long-term correlation window

        Returns:
            Dictionary with correlation change analysis
        """
        short_corr = self.calculate_rolling_correlation(series1, series2, short_window)
        long_corr = self.calculate_rolling_correlation(series1, series2, long_window)

        if len(short_corr) < 2 or len(long_corr) < 2:
            return {
                "regime_change": False,
                "short_term_corr": 0,
                "long_term_corr": 0,
                "divergence": 0
            }

        current_short = short_corr.iloc[-1] if not pd.isna(short_corr.iloc[-1]) else 0
        current_long = long_corr.iloc[-1] if not pd.isna(long_corr.iloc[-1]) else 0
        divergence = current_short - current_long

        # Regime change if divergence is significant
        regime_change = abs(divergence) > 0.3

        return {
            "regime_change": regime_change,
            "short_term_corr": round(current_short, 4),
            "long_term_corr": round(current_long, 4),
            "divergence": round(divergence, 4),
            "interpretation": self._interpret_divergence(divergence)
        }

    def _interpret_divergence(self, divergence: float) -> str:
        """Interpret correlation divergence."""
        if divergence > 0.3:
            return "Short-term correlation strengthening - assets moving together recently"
        elif divergence < -0.3:
            return "Short-term correlation weakening - assets diverging recently"
        else:
            return "Correlation regime stable"

    def matrix_to_dict(self, correlation_matrix: CorrelationMatrix) -> Dict:
        """
        Convert CorrelationMatrix to dictionary for JSON serialization.

        Args:
            correlation_matrix: CorrelationMatrix object

        Returns:
            Dictionary representation
        """
        return {
            "assets": correlation_matrix.assets,
            "matrix": correlation_matrix.matrix.to_dict(),
            "period": correlation_matrix.period,
            "timestamp": correlation_matrix.timestamp,
            "strongest_positive": {
                "assets": [correlation_matrix.strongest_positive.asset1,
                          correlation_matrix.strongest_positive.asset2],
                "correlation": correlation_matrix.strongest_positive.correlation,
                "strength": correlation_matrix.strongest_positive.strength.value
            } if correlation_matrix.strongest_positive else None,
            "strongest_negative": {
                "assets": [correlation_matrix.strongest_negative.asset1,
                          correlation_matrix.strongest_negative.asset2],
                "correlation": correlation_matrix.strongest_negative.correlation,
                "strength": correlation_matrix.strongest_negative.strength.value
            } if correlation_matrix.strongest_negative else None
        }
