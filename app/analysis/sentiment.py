"""Sentiment analysis for news and social media content."""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime


class SentimentLabel(Enum):
    """Sentiment classification labels."""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class SentimentResult:
    """Result of sentiment analysis on a single text."""
    text: str
    score: float  # -1.0 to 1.0
    label: SentimentLabel
    confidence: float  # 0.0 to 1.0
    keywords_found: List[str]


@dataclass
class AggregateSentiment:
    """Aggregated sentiment for a collection of texts."""
    source: str
    total_analyzed: int
    avg_score: float
    label: SentimentLabel
    distribution: Dict[str, int]
    top_bullish_keywords: List[Tuple[str, int]]
    top_bearish_keywords: List[Tuple[str, int]]
    trend: str  # 'improving', 'declining', 'stable'
    timestamp: datetime


class SentimentAnalyzer:
    """
    Analyzes sentiment in financial news and social media content.

    Uses lexicon-based approach with financial-specific keywords,
    enhanced with VADER sentiment when available.
    """

    # Bullish keywords and phrases
    BULLISH_KEYWORDS = {
        # Strong bullish
        "moon": 2.0, "mooning": 2.0, "rocket": 1.8, "bullish": 1.5,
        "breakout": 1.5, "surge": 1.5, "soar": 1.5, "skyrocket": 1.8,
        "all-time high": 1.8, "ath": 1.5, "rally": 1.3, "boom": 1.5,

        # Moderate bullish
        "buy": 0.8, "long": 0.7, "growth": 0.8, "profit": 0.8,
        "gains": 0.8, "winning": 0.7, "beat": 0.6, "exceed": 0.7,
        "upgrade": 0.8, "outperform": 0.8, "strong": 0.6, "positive": 0.6,
        "optimistic": 0.7, "confidence": 0.6, "recovery": 0.7,

        # Mild bullish
        "up": 0.4, "rise": 0.4, "increase": 0.4, "higher": 0.4,
        "support": 0.3, "accumulate": 0.5, "hold": 0.2
    }

    # Bearish keywords and phrases
    BEARISH_KEYWORDS = {
        # Strong bearish
        "crash": -2.0, "collapse": -2.0, "plunge": -1.8, "dump": -1.5,
        "bearish": -1.5, "capitulation": -1.8, "panic": -1.5,
        "bloodbath": -1.8, "rekt": -1.5, "liquidation": -1.5,

        # Moderate bearish
        "sell": -0.8, "short": -0.7, "loss": -0.8, "losing": -0.7,
        "decline": -0.7, "fall": -0.6, "drop": -0.6, "downgrade": -0.8,
        "underperform": -0.8, "weak": -0.6, "negative": -0.6,
        "pessimistic": -0.7, "fear": -0.8, "concern": -0.5,

        # Mild bearish
        "down": -0.4, "lower": -0.4, "decrease": -0.4, "resistance": -0.3,
        "caution": -0.3, "risk": -0.4, "uncertain": -0.3
    }

    # Intensifiers and negations
    INTENSIFIERS = {
        "very": 1.5, "extremely": 2.0, "incredibly": 1.8,
        "super": 1.5, "massive": 1.5, "huge": 1.5
    }

    NEGATIONS = {"not", "no", "never", "neither", "nobody", "nothing",
                 "nowhere", "hardly", "barely", "doesn't", "isn't",
                 "wasn't", "shouldn't", "wouldn't", "couldn't", "won't"}

    def __init__(self, use_vader: bool = True):
        """
        Initialize sentiment analyzer.

        Args:
            use_vader: Whether to use VADER sentiment as fallback
        """
        self.use_vader = use_vader
        self._vader = None

        if use_vader:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
            except ImportError:
                self.use_vader = False

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove mentions and hashtags (but keep the text)
        text = re.sub(r'[@#](\w+)', r'\1', text)

        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text

    def _find_keywords(
        self,
        text: str
    ) -> Tuple[List[str], List[str], float]:
        """
        Find bullish and bearish keywords in text.

        Returns:
            Tuple of (bullish_keywords, bearish_keywords, raw_score)
        """
        words = text.split()
        bullish_found = []
        bearish_found = []
        score = 0.0

        # Check for negation in context
        negation_window = 3  # words to look back for negation

        for i, word in enumerate(words):
            # Check for negation before this word
            negated = any(
                words[j] in self.NEGATIONS
                for j in range(max(0, i - negation_window), i)
            )

            # Check for intensifier before this word
            intensifier = 1.0
            for j in range(max(0, i - 2), i):
                if words[j] in self.INTENSIFIERS:
                    intensifier = self.INTENSIFIERS[words[j]]
                    break

            # Check bullish keywords
            if word in self.BULLISH_KEYWORDS:
                keyword_score = self.BULLISH_KEYWORDS[word] * intensifier
                if negated:
                    keyword_score = -keyword_score
                    bearish_found.append(word)
                else:
                    bullish_found.append(word)
                score += keyword_score

            # Check bearish keywords
            elif word in self.BEARISH_KEYWORDS:
                keyword_score = self.BEARISH_KEYWORDS[word] * intensifier
                if negated:
                    keyword_score = -keyword_score
                    bullish_found.append(word)
                else:
                    bearish_found.append(word)
                score += keyword_score

        return bullish_found, bearish_found, score

    def _score_to_label(self, score: float) -> SentimentLabel:
        """Convert sentiment score to label."""
        if score >= 0.6:
            return SentimentLabel.VERY_BULLISH
        elif score >= 0.2:
            return SentimentLabel.BULLISH
        elif score >= -0.2:
            return SentimentLabel.NEUTRAL
        elif score >= -0.6:
            return SentimentLabel.BEARISH
        else:
            return SentimentLabel.VERY_BEARISH

    def analyze_text(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult object
        """
        processed = self._preprocess_text(text)
        bullish, bearish, lexicon_score = self._find_keywords(processed)

        # Use VADER if available
        vader_score = 0.0
        if self._vader:
            vader_result = self._vader.polarity_scores(text)
            vader_score = vader_result['compound']

        # Combine scores (weighted average)
        if self._vader and (bullish or bearish):
            # Weight lexicon more if we found financial keywords
            final_score = 0.6 * (lexicon_score / 5) + 0.4 * vader_score
        elif self._vader:
            # Use VADER primarily if no keywords found
            final_score = vader_score
        else:
            # Normalize lexicon score to -1 to 1
            final_score = max(-1, min(1, lexicon_score / 5))

        # Calculate confidence based on signal strength
        keywords_found = len(bullish) + len(bearish)
        confidence = min(1.0, 0.3 + (keywords_found * 0.15) + abs(final_score) * 0.3)

        return SentimentResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            score=round(final_score, 4),
            label=self._score_to_label(final_score),
            confidence=round(confidence, 2),
            keywords_found=bullish + bearish
        )

    def analyze_batch(
        self,
        texts: List[str],
        source: str = "unknown"
    ) -> AggregateSentiment:
        """
        Analyze sentiment of multiple texts.

        Args:
            texts: List of texts to analyze
            source: Source identifier (e.g., 'news', 'twitter')

        Returns:
            AggregateSentiment object
        """
        if not texts:
            return AggregateSentiment(
                source=source,
                total_analyzed=0,
                avg_score=0.0,
                label=SentimentLabel.NEUTRAL,
                distribution={"very_bullish": 0, "bullish": 0, "neutral": 0,
                            "bearish": 0, "very_bearish": 0},
                top_bullish_keywords=[],
                top_bearish_keywords=[],
                trend="stable",
                timestamp=datetime.now()
            )

        results = [self.analyze_text(text) for text in texts]

        # Calculate distribution
        distribution = {"very_bullish": 0, "bullish": 0, "neutral": 0,
                       "bearish": 0, "very_bearish": 0}
        for r in results:
            distribution[r.label.value] += 1

        # Calculate average score
        avg_score = sum(r.score for r in results) / len(results)

        # Count keywords
        bullish_counts: Dict[str, int] = {}
        bearish_counts: Dict[str, int] = {}

        for r in results:
            for kw in r.keywords_found:
                if kw in self.BULLISH_KEYWORDS:
                    bullish_counts[kw] = bullish_counts.get(kw, 0) + 1
                elif kw in self.BEARISH_KEYWORDS:
                    bearish_counts[kw] = bearish_counts.get(kw, 0) + 1

        top_bullish = sorted(bullish_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_bearish = sorted(bearish_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Determine trend (would need historical data for real trend)
        bullish_ratio = (distribution["very_bullish"] + distribution["bullish"]) / len(results)
        if bullish_ratio > 0.6:
            trend = "improving"
        elif bullish_ratio < 0.4:
            trend = "declining"
        else:
            trend = "stable"

        return AggregateSentiment(
            source=source,
            total_analyzed=len(results),
            avg_score=round(avg_score, 4),
            label=self._score_to_label(avg_score),
            distribution=distribution,
            top_bullish_keywords=top_bullish,
            top_bearish_keywords=top_bearish,
            trend=trend,
            timestamp=datetime.now()
        )

    def get_market_sentiment(
        self,
        news_texts: List[str] = None,
        social_texts: List[str] = None,
        weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Get combined market sentiment from multiple sources.

        Args:
            news_texts: List of news article texts
            social_texts: List of social media texts
            weights: Weight for each source (default: news=0.6, social=0.4)

        Returns:
            Dictionary with combined sentiment analysis
        """
        weights = weights or {"news": 0.6, "social": 0.4}
        news_texts = news_texts or []
        social_texts = social_texts or []

        news_sentiment = self.analyze_batch(news_texts, "news") if news_texts else None
        social_sentiment = self.analyze_batch(social_texts, "social") if social_texts else None

        # Calculate weighted average
        total_weight = 0.0
        weighted_score = 0.0

        if news_sentiment and news_sentiment.total_analyzed > 0:
            weighted_score += news_sentiment.avg_score * weights["news"]
            total_weight += weights["news"]

        if social_sentiment and social_sentiment.total_analyzed > 0:
            weighted_score += social_sentiment.avg_score * weights["social"]
            total_weight += weights["social"]

        combined_score = weighted_score / total_weight if total_weight > 0 else 0.0

        return {
            "combined_score": round(combined_score, 4),
            "combined_label": self._score_to_label(combined_score).value,
            "news": {
                "score": news_sentiment.avg_score if news_sentiment else None,
                "label": news_sentiment.label.value if news_sentiment else None,
                "count": news_sentiment.total_analyzed if news_sentiment else 0,
                "distribution": news_sentiment.distribution if news_sentiment else None
            },
            "social": {
                "score": social_sentiment.avg_score if social_sentiment else None,
                "label": social_sentiment.label.value if social_sentiment else None,
                "count": social_sentiment.total_analyzed if social_sentiment else 0,
                "distribution": social_sentiment.distribution if social_sentiment else None
            },
            "timestamp": datetime.now().isoformat()
        }

    def sentiment_to_dict(self, sentiment: AggregateSentiment) -> Dict:
        """Convert AggregateSentiment to dictionary."""
        return {
            "source": sentiment.source,
            "total_analyzed": sentiment.total_analyzed,
            "avg_score": sentiment.avg_score,
            "label": sentiment.label.value,
            "distribution": sentiment.distribution,
            "top_bullish_keywords": sentiment.top_bullish_keywords,
            "top_bearish_keywords": sentiment.top_bearish_keywords,
            "trend": sentiment.trend,
            "timestamp": sentiment.timestamp.isoformat()
        }
