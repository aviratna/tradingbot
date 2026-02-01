"""Tests for sentiment analysis module."""
import pytest
from datetime import datetime

from app.analysis.sentiment import (
    SentimentAnalyzer,
    SentimentResult,
    AggregateSentiment,
    SentimentLabel
)


@pytest.fixture
def analyzer():
    """Create a sentiment analyzer instance."""
    return SentimentAnalyzer(use_vader=True)


@pytest.fixture
def analyzer_no_vader():
    """Create a sentiment analyzer without VADER."""
    return SentimentAnalyzer(use_vader=False)


class TestSentimentLabel:
    """Tests for SentimentLabel enum."""

    def test_label_values(self):
        """Test sentiment label values."""
        assert SentimentLabel.VERY_BULLISH.value == "very_bullish"
        assert SentimentLabel.BULLISH.value == "bullish"
        assert SentimentLabel.NEUTRAL.value == "neutral"
        assert SentimentLabel.BEARISH.value == "bearish"
        assert SentimentLabel.VERY_BEARISH.value == "very_bearish"


class TestTextPreprocessing:
    """Tests for text preprocessing."""

    def test_lowercase_conversion(self, analyzer):
        """Test text is converted to lowercase."""
        processed = analyzer._preprocess_text("BULLISH on BITCOIN!")
        assert processed == processed.lower()

    def test_url_removal(self, analyzer):
        """Test URLs are removed."""
        text = "Check this out https://example.com and http://test.com"
        processed = analyzer._preprocess_text(text)
        assert "https" not in processed
        assert "http" not in processed

    def test_mentions_cleaned(self, analyzer):
        """Test @mentions are cleaned."""
        text = "@trader says $BTC is mooning #crypto"
        processed = analyzer._preprocess_text(text)
        assert "@" not in processed
        assert "#" not in processed

    def test_whitespace_normalized(self, analyzer):
        """Test whitespace is normalized."""
        text = "too   many    spaces"
        processed = analyzer._preprocess_text(text)
        assert "   " not in processed


class TestKeywordDetection:
    """Tests for bullish/bearish keyword detection."""

    def test_find_bullish_keywords(self, analyzer):
        """Test finding bullish keywords."""
        text = "bitcoin is mooning bullish breakout"
        bullish, bearish, score = analyzer._find_keywords(text)

        assert len(bullish) > 0
        assert "mooning" in bullish or "bullish" in bullish
        assert score > 0

    def test_find_bearish_keywords(self, analyzer):
        """Test finding bearish keywords."""
        text = "market crash panic selling dump"
        bullish, bearish, score = analyzer._find_keywords(text)

        assert len(bearish) > 0
        assert score < 0

    def test_negation_handling(self, analyzer):
        """Test that negation inverts sentiment."""
        text_positive = "this is bullish"
        text_negated = "this is not bullish"

        _, _, score_positive = analyzer._find_keywords(text_positive)
        _, _, score_negated = analyzer._find_keywords(text_negated)

        # Negated should be opposite or less positive
        assert score_negated < score_positive

    def test_intensifier_handling(self, analyzer):
        """Test that intensifiers amplify sentiment."""
        text_normal = "this is bullish"
        text_intense = "this is very bullish"

        _, _, score_normal = analyzer._find_keywords(text_normal)
        _, _, score_intense = analyzer._find_keywords(text_intense)

        # Intensified should be stronger
        assert score_intense >= score_normal

    def test_no_keywords_zero_score(self, analyzer):
        """Test that text without keywords has zero score."""
        text = "the weather is nice today"
        bullish, bearish, score = analyzer._find_keywords(text)

        assert len(bullish) == 0
        assert len(bearish) == 0
        assert score == 0


class TestScoreToLabel:
    """Tests for score to label conversion."""

    def test_very_bullish_threshold(self, analyzer):
        """Test very bullish threshold."""
        assert analyzer._score_to_label(0.7) == SentimentLabel.VERY_BULLISH
        assert analyzer._score_to_label(1.0) == SentimentLabel.VERY_BULLISH

    def test_bullish_threshold(self, analyzer):
        """Test bullish threshold."""
        assert analyzer._score_to_label(0.3) == SentimentLabel.BULLISH
        assert analyzer._score_to_label(0.5) == SentimentLabel.BULLISH

    def test_neutral_threshold(self, analyzer):
        """Test neutral threshold."""
        assert analyzer._score_to_label(0.0) == SentimentLabel.NEUTRAL
        assert analyzer._score_to_label(0.1) == SentimentLabel.NEUTRAL
        assert analyzer._score_to_label(-0.1) == SentimentLabel.NEUTRAL

    def test_bearish_threshold(self, analyzer):
        """Test bearish threshold."""
        assert analyzer._score_to_label(-0.3) == SentimentLabel.BEARISH
        assert analyzer._score_to_label(-0.5) == SentimentLabel.BEARISH

    def test_very_bearish_threshold(self, analyzer):
        """Test very bearish threshold."""
        assert analyzer._score_to_label(-0.7) == SentimentLabel.VERY_BEARISH
        assert analyzer._score_to_label(-1.0) == SentimentLabel.VERY_BEARISH


class TestAnalyzeText:
    """Tests for single text analysis."""

    def test_analyze_bullish_text(self, analyzer):
        """Test analyzing bullish text."""
        text = "Bitcoin is mooning! Very bullish on crypto! Buy the dip!"
        result = analyzer.analyze_text(text)

        assert isinstance(result, SentimentResult)
        assert result.score > 0
        assert result.label in [SentimentLabel.BULLISH, SentimentLabel.VERY_BULLISH]
        assert len(result.keywords_found) > 0

    def test_analyze_bearish_text(self, analyzer):
        """Test analyzing bearish text."""
        text = "Market crash incoming! Panic selling everywhere. Total collapse."
        result = analyzer.analyze_text(text)

        assert isinstance(result, SentimentResult)
        assert result.score < 0
        assert result.label in [SentimentLabel.BEARISH, SentimentLabel.VERY_BEARISH]

    def test_analyze_neutral_text(self, analyzer):
        """Test analyzing neutral text."""
        text = "The market opened today. Trading volume was average."
        result = analyzer.analyze_text(text)

        assert isinstance(result, SentimentResult)
        assert -0.3 <= result.score <= 0.3

    def test_confidence_increases_with_keywords(self, analyzer):
        """Test that confidence increases with more keywords."""
        text_few = "bullish"
        text_many = "very bullish moon rocket gains breakout rally"

        result_few = analyzer.analyze_text(text_few)
        result_many = analyzer.analyze_text(text_many)

        assert result_many.confidence >= result_few.confidence

    def test_text_truncation_in_result(self, analyzer):
        """Test that long text is truncated in result."""
        long_text = "bullish " * 100
        result = analyzer.analyze_text(long_text)

        assert len(result.text) <= 103  # 100 chars + "..."


class TestAnalyzeBatch:
    """Tests for batch text analysis."""

    def test_analyze_batch_basic(self, analyzer):
        """Test basic batch analysis."""
        texts = [
            "Very bullish on this!",
            "Bearish outlook for the market",
            "Just a normal update"
        ]

        result = analyzer.analyze_batch(texts, source="test")

        assert isinstance(result, AggregateSentiment)
        assert result.total_analyzed == 3
        assert result.source == "test"

    def test_analyze_batch_empty(self, analyzer):
        """Test batch analysis with empty list."""
        result = analyzer.analyze_batch([], source="test")

        assert result.total_analyzed == 0
        assert result.avg_score == 0.0
        assert result.label == SentimentLabel.NEUTRAL

    def test_distribution_counts(self, analyzer):
        """Test that distribution counts are accurate."""
        texts = [
            "Very very bullish moon rocket!",
            "Bullish outlook",
            "Neutral stance",
            "Bearish trend",
            "Total crash panic dump!"
        ]

        result = analyzer.analyze_batch(texts, source="test")

        total_dist = sum(result.distribution.values())
        assert total_dist == 5

    def test_top_keywords_extracted(self, analyzer):
        """Test that top keywords are extracted."""
        texts = [
            "Bullish bullish bullish!",
            "Moon moon moon!",
            "Buy buy buy!"
        ]

        result = analyzer.analyze_batch(texts, source="test")

        assert len(result.top_bullish_keywords) > 0
        # Most frequent should be first
        if len(result.top_bullish_keywords) > 1:
            assert result.top_bullish_keywords[0][1] >= result.top_bullish_keywords[1][1]

    def test_trend_calculation(self, analyzer):
        """Test trend calculation."""
        bullish_texts = ["Bullish!" for _ in range(10)]
        bearish_texts = ["Bearish crash" for _ in range(10)]
        mixed_texts = ["Bullish", "Bearish", "Neutral", "Bullish", "Bearish"]

        bullish_result = analyzer.analyze_batch(bullish_texts)
        bearish_result = analyzer.analyze_batch(bearish_texts)
        mixed_result = analyzer.analyze_batch(mixed_texts)

        assert bullish_result.trend == "improving"
        assert bearish_result.trend == "declining"
        assert mixed_result.trend == "stable"


class TestMarketSentiment:
    """Tests for combined market sentiment analysis."""

    def test_market_sentiment_combined(self, analyzer):
        """Test combined market sentiment."""
        news = ["Stock market rally continues", "Earnings beat expectations"]
        social = ["Bullish on $AAPL!", "Moon incoming!"]

        result = analyzer.get_market_sentiment(news, social)

        assert isinstance(result, dict)
        assert "combined_score" in result
        assert "combined_label" in result
        assert "news" in result
        assert "social" in result

    def test_market_sentiment_news_only(self, analyzer):
        """Test market sentiment with only news."""
        news = ["Stock market rally", "Strong earnings"]

        result = analyzer.get_market_sentiment(news_texts=news)

        assert result["news"]["count"] == 2
        assert result["social"]["count"] == 0

    def test_market_sentiment_social_only(self, analyzer):
        """Test market sentiment with only social."""
        social = ["Bullish!", "Moon!"]

        result = analyzer.get_market_sentiment(social_texts=social)

        assert result["news"]["count"] == 0
        assert result["social"]["count"] == 2

    def test_market_sentiment_weights(self, analyzer):
        """Test that weights affect combined score."""
        # All bullish news, all bearish social
        news = ["Very bullish rally"] * 5
        social = ["Crash dump panic"] * 5

        # Weight news higher
        result_news = analyzer.get_market_sentiment(
            news, social, weights={"news": 0.9, "social": 0.1}
        )

        # Weight social higher
        result_social = analyzer.get_market_sentiment(
            news, social, weights={"news": 0.1, "social": 0.9}
        )

        assert result_news["combined_score"] > result_social["combined_score"]


class TestSentimentSerialization:
    """Tests for sentiment serialization."""

    def test_sentiment_to_dict(self, analyzer):
        """Test converting sentiment to dictionary."""
        texts = ["Bullish!", "Bearish", "Neutral"]
        sentiment = analyzer.analyze_batch(texts, source="test")
        result = analyzer.sentiment_to_dict(sentiment)

        assert isinstance(result, dict)
        assert "source" in result
        assert "total_analyzed" in result
        assert "avg_score" in result
        assert "label" in result
        assert "distribution" in result
        assert "timestamp" in result

    def test_dict_is_json_serializable(self, analyzer):
        """Test that sentiment dict can be serialized to JSON."""
        import json

        texts = ["Bullish!", "Bearish"]
        sentiment = analyzer.analyze_batch(texts)
        result = analyzer.sentiment_to_dict(sentiment)

        json_str = json.dumps(result)
        assert isinstance(json_str, str)


class TestWithoutVADER:
    """Tests for sentiment analysis without VADER."""

    def test_lexicon_only_analysis(self, analyzer_no_vader):
        """Test analysis works without VADER."""
        text = "Very bullish on crypto! Moon incoming!"
        result = analyzer_no_vader.analyze_text(text)

        assert isinstance(result, SentimentResult)
        assert result.score > 0

    def test_batch_without_vader(self, analyzer_no_vader):
        """Test batch analysis without VADER."""
        texts = ["Bullish!", "Bearish", "Neutral"]
        result = analyzer_no_vader.analyze_batch(texts)

        assert result.total_analyzed == 3
