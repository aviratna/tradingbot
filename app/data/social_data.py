"""Social media data fetcher for sentiment analysis from X/Twitter."""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import httpx
from ..config import settings


@dataclass
class SocialPost:
    """Social media post data structure."""
    id: str
    text: str
    author: str
    created_at: datetime
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    sentiment_score: float = 0.0


@dataclass
class SocialSentiment:
    """Aggregated social sentiment data."""
    topic: str
    posts: List[SocialPost]
    total_posts: int
    avg_sentiment: float
    sentiment_distribution: Dict[str, int]  # positive, negative, neutral counts
    trending_score: float
    last_updated: datetime


class SocialDataFetcher:
    """Fetches social media data and sentiment from X/Twitter."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self.twitter_api_url = "https://api.twitter.com/2"

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache:
            return False
        age = (datetime.now() - self._cache_timestamps.get(key, datetime.min)).total_seconds()
        return age < settings.CACHE_TTL_SOCIAL

    def _set_cache(self, key: str, value: Any) -> None:
        """Set cached data."""
        self._cache[key] = value
        self._cache_timestamps[key] = datetime.now()

    def _get_cache(self, key: str) -> Any:
        """Get cached data."""
        return self._cache.get(key)

    async def fetch_twitter_posts(
        self,
        query: str,
        max_results: int = 100
    ) -> List[SocialPost]:
        """
        Fetch posts from X/Twitter API.

        Args:
            query: Search query
            max_results: Maximum number of posts to fetch

        Returns:
            List of SocialPost objects
        """
        if not settings.TWITTER_BEARER_TOKEN:
            return self._generate_mock_posts(query)

        cache_key = f"twitter_{query}"
        if self._is_cache_valid(cache_key):
            return self._get_cache(cache_key)

        posts = []

        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {settings.TWITTER_BEARER_TOKEN}"
                }
                url = f"{self.twitter_api_url}/tweets/search/recent"
                params = {
                    "query": f"{query} lang:en -is:retweet",
                    "max_results": min(max_results, 100),
                    "tweet.fields": "created_at,public_metrics,author_id",
                    "expansions": "author_id"
                }

                resp = await client.get(url, headers=headers, params=params, timeout=10)
                data = resp.json()

                # Build author map
                authors = {}
                for user in data.get('includes', {}).get('users', []):
                    authors[user['id']] = user['username']

                for tweet in data.get('data', []):
                    metrics = tweet.get('public_metrics', {})
                    try:
                        created_at = datetime.fromisoformat(
                            tweet['created_at'].replace('Z', '+00:00')
                        )
                    except (ValueError, TypeError):
                        created_at = datetime.now()

                    post = SocialPost(
                        id=tweet['id'],
                        text=tweet['text'],
                        author=authors.get(tweet.get('author_id'), 'unknown'),
                        created_at=created_at,
                        likes=metrics.get('like_count', 0),
                        retweets=metrics.get('retweet_count', 0),
                        replies=metrics.get('reply_count', 0)
                    )
                    posts.append(post)

                self._set_cache(cache_key, posts)

        except Exception as e:
            print(f"Error fetching Twitter data: {e}")
            posts = self._generate_mock_posts(query)

        return posts

    def _generate_mock_posts(self, query: str) -> List[SocialPost]:
        """
        Generate mock social posts for demo purposes.
        Used when Twitter API is not available.
        """
        import random

        sentiments = [
            ("Bullish on ${query}! Looking great for the long term.", 0.8),
            ("${query} showing strong momentum today!", 0.7),
            ("Not sure about ${query}, market seems uncertain.", 0.0),
            ("${query} is the future, definitely adding more.", 0.9),
            ("Taking profits on ${query}, been a good run.", 0.3),
            ("${query} breaking out! This is huge!", 0.85),
            ("Cautious on ${query} with current macro conditions.", -0.2),
            ("${query} consolidating, expecting a move soon.", 0.1),
            ("Love the fundamentals of ${query}.", 0.6),
            ("${query} looking weak, might wait for a dip.", -0.3),
        ]

        posts = []
        for i, (text_template, sentiment) in enumerate(sentiments):
            text = text_template.replace("${query}", query)
            posts.append(SocialPost(
                id=f"mock_{i}_{query}",
                text=text,
                author=f"trader_{random.randint(1000, 9999)}",
                created_at=datetime.now() - timedelta(hours=random.randint(0, 24)),
                likes=random.randint(10, 1000),
                retweets=random.randint(1, 100),
                replies=random.randint(0, 50),
                sentiment_score=sentiment
            ))

        return posts

    def _calculate_sentiment_distribution(
        self,
        posts: List[SocialPost]
    ) -> Dict[str, int]:
        """Calculate sentiment distribution from posts."""
        distribution = {"positive": 0, "negative": 0, "neutral": 0}

        for post in posts:
            if post.sentiment_score > 0.2:
                distribution["positive"] += 1
            elif post.sentiment_score < -0.2:
                distribution["negative"] += 1
            else:
                distribution["neutral"] += 1

        return distribution

    def _calculate_trending_score(self, posts: List[SocialPost]) -> float:
        """
        Calculate trending score based on engagement.

        Higher engagement (likes, retweets, replies) = higher trending score.
        """
        if not posts:
            return 0.0

        total_engagement = sum(
            p.likes + (p.retweets * 2) + (p.replies * 1.5)
            for p in posts
        )

        # Normalize to 0-100 scale
        avg_engagement = total_engagement / len(posts)
        return min(100, avg_engagement / 10)

    async def get_asset_sentiment(self, asset: str) -> SocialSentiment:
        """
        Get social sentiment for a specific asset.

        Args:
            asset: Asset symbol or name (e.g., 'AAPL', 'bitcoin', '$BTC')

        Returns:
            SocialSentiment object with aggregated data
        """
        # Add common financial hashtags/cashtags
        query = f"{asset} OR #{asset} OR ${asset}"
        posts = await self.fetch_twitter_posts(query)

        # Calculate average sentiment
        avg_sentiment = 0.0
        if posts:
            avg_sentiment = sum(p.sentiment_score for p in posts) / len(posts)

        return SocialSentiment(
            topic=asset,
            posts=posts,
            total_posts=len(posts),
            avg_sentiment=avg_sentiment,
            sentiment_distribution=self._calculate_sentiment_distribution(posts),
            trending_score=self._calculate_trending_score(posts),
            last_updated=datetime.now()
        )

    async def get_market_sentiment(
        self,
        market_type: str = "all"
    ) -> SocialSentiment:
        """
        Get social sentiment for a market type.

        Args:
            market_type: One of 'stocks', 'crypto', 'forex', 'all'

        Returns:
            SocialSentiment object with aggregated data
        """
        queries = {
            "stocks": "stock market OR #stocks OR wall street",
            "crypto": "crypto OR bitcoin OR ethereum OR #crypto",
            "forex": "forex OR currency OR #forex",
            "all": "financial markets OR trading OR investing"
        }

        query = queries.get(market_type, queries["all"])
        posts = await self.fetch_twitter_posts(query)

        avg_sentiment = 0.0
        if posts:
            avg_sentiment = sum(p.sentiment_score for p in posts) / len(posts)

        return SocialSentiment(
            topic=market_type,
            posts=posts,
            total_posts=len(posts),
            avg_sentiment=avg_sentiment,
            sentiment_distribution=self._calculate_sentiment_distribution(posts),
            trending_score=self._calculate_trending_score(posts),
            last_updated=datetime.now()
        )

    async def get_all_sentiments(
        self,
        assets: List[str] = None
    ) -> Dict[str, SocialSentiment]:
        """
        Get social sentiment for multiple assets.

        Args:
            assets: List of asset symbols/names

        Returns:
            Dictionary mapping assets to SocialSentiment
        """
        assets = assets or ["bitcoin", "ethereum", "AAPL", "SPY"]
        tasks = [self.get_asset_sentiment(asset) for asset in assets]
        results = await asyncio.gather(*tasks)

        return {asset: result for asset, result in zip(assets, results)}
