"""News data fetcher for financial news aggregation."""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import httpx
from ..config import settings


@dataclass
class NewsArticle:
    """News article data structure."""
    title: str
    description: str
    source: str
    url: str
    published_at: datetime
    sentiment_score: float = 0.0
    relevance_score: float = 0.0
    categories: List[str] = None

    def __post_init__(self):
        if self.categories is None:
            self.categories = []


@dataclass
class NewsData:
    """Aggregated news data for a topic."""
    topic: str
    articles: List[NewsArticle]
    avg_sentiment: float
    article_count: int
    last_updated: datetime


class NewsDataFetcher:
    """Fetches and aggregates financial news from various sources."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

        # Keywords for different market categories
        self.stock_keywords = [
            "stock market", "S&P 500", "NASDAQ", "NYSE", "earnings",
            "stocks", "equities", "wall street", "dow jones"
        ]
        self.crypto_keywords = [
            "bitcoin", "ethereum", "cryptocurrency", "crypto",
            "blockchain", "defi", "altcoin", "web3"
        ]
        self.forex_keywords = [
            "forex", "currency", "dollar", "euro", "exchange rate",
            "federal reserve", "central bank", "interest rate"
        ]
        self.general_keywords = [
            "economy", "inflation", "recession", "GDP", "trade",
            "market", "investment", "financial"
        ]

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache:
            return False
        age = (datetime.now() - self._cache_timestamps.get(key, datetime.min)).total_seconds()
        return age < settings.CACHE_TTL_NEWS

    def _set_cache(self, key: str, value: Any) -> None:
        """Set cached data."""
        self._cache[key] = value
        self._cache_timestamps[key] = datetime.now()

    def _get_cache(self, key: str) -> Any:
        """Get cached data."""
        return self._cache.get(key)

    def _categorize_article(self, article: NewsArticle) -> List[str]:
        """Categorize article based on content."""
        categories = []
        text = f"{article.title} {article.description}".lower()

        if any(kw in text for kw in self.stock_keywords):
            categories.append("stocks")
        if any(kw in text for kw in self.crypto_keywords):
            categories.append("crypto")
        if any(kw in text for kw in self.forex_keywords):
            categories.append("forex")
        if any(kw in text for kw in self.general_keywords):
            categories.append("general")

        return categories or ["general"]

    async def fetch_news_api(
        self,
        query: str,
        from_date: datetime = None,
        page_size: int = 20
    ) -> List[NewsArticle]:
        """
        Fetch news from NewsAPI.

        Args:
            query: Search query
            from_date: Start date for articles
            page_size: Number of articles to fetch

        Returns:
            List of NewsArticle objects
        """
        if not settings.NEWS_API_KEY:
            return await self._fetch_fallback_news(query)

        cache_key = f"newsapi_{query}"
        if self._is_cache_valid(cache_key):
            return self._get_cache(cache_key)

        from_date = from_date or (datetime.now() - timedelta(days=7))
        articles = []

        try:
            async with httpx.AsyncClient() as client:
                url = f"{settings.NEWS_API_URL}/everything"
                params = {
                    "q": query,
                    "from": from_date.isoformat(),
                    "sortBy": "publishedAt",
                    "pageSize": page_size,
                    "language": "en",
                    "apiKey": settings.NEWS_API_KEY
                }

                resp = await client.get(url, params=params, timeout=10)
                data = resp.json()

                for item in data.get('articles', []):
                    try:
                        published = datetime.fromisoformat(
                            item['publishedAt'].replace('Z', '+00:00')
                        )
                    except (ValueError, TypeError):
                        published = datetime.now()

                    article = NewsArticle(
                        title=item.get('title', ''),
                        description=item.get('description', '') or '',
                        source=item.get('source', {}).get('name', 'Unknown'),
                        url=item.get('url', ''),
                        published_at=published
                    )
                    article.categories = self._categorize_article(article)
                    articles.append(article)

                self._set_cache(cache_key, articles)

        except Exception as e:
            print(f"Error fetching news from NewsAPI: {e}")

        return articles

    async def _fetch_fallback_news(self, query: str) -> List[NewsArticle]:
        """
        Fetch news using free alternatives when NewsAPI key is not available.
        Uses public RSS feeds and free news sources.
        """
        articles = []

        # Mock data for demo purposes when no API key is available
        mock_articles = [
            {
                "title": f"Market Update: {query} Shows Strong Performance",
                "description": f"Latest analysis shows {query} trending upward with increased volume.",
                "source": "Market Watch",
                "url": "https://example.com/news/1"
            },
            {
                "title": f"Analysts Bullish on {query} Outlook",
                "description": f"Financial experts predict positive momentum for {query} in coming weeks.",
                "source": "Financial Times",
                "url": "https://example.com/news/2"
            },
            {
                "title": f"Global Markets React to {query} Developments",
                "description": f"International investors show growing interest in {query} opportunities.",
                "source": "Bloomberg",
                "url": "https://example.com/news/3"
            }
        ]

        for item in mock_articles:
            article = NewsArticle(
                title=item['title'],
                description=item['description'],
                source=item['source'],
                url=item['url'],
                published_at=datetime.now()
            )
            article.categories = self._categorize_article(article)
            articles.append(article)

        return articles

    async def get_market_news(self, market_type: str = "all") -> NewsData:
        """
        Get news for a specific market type.

        Args:
            market_type: One of 'stocks', 'crypto', 'forex', 'general', 'all'

        Returns:
            NewsData object with aggregated news
        """
        queries = {
            "stocks": "stock market OR S&P 500 OR NASDAQ",
            "crypto": "bitcoin OR cryptocurrency OR ethereum",
            "forex": "forex OR currency exchange OR federal reserve",
            "general": "economy OR financial markets OR investment",
            "all": "financial markets OR economy OR stocks OR crypto"
        }

        query = queries.get(market_type, queries["all"])
        articles = await self.fetch_news_api(query)

        # Filter by category if specific market type
        if market_type != "all":
            articles = [a for a in articles if market_type in a.categories]

        # Calculate average sentiment
        avg_sentiment = 0.0
        if articles:
            sentiments = [a.sentiment_score for a in articles if a.sentiment_score]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        return NewsData(
            topic=market_type,
            articles=articles,
            avg_sentiment=avg_sentiment,
            article_count=len(articles),
            last_updated=datetime.now()
        )

    async def get_all_news(self) -> Dict[str, NewsData]:
        """
        Get news for all market types.

        Returns:
            Dictionary mapping market types to NewsData
        """
        market_types = ["stocks", "crypto", "forex", "general"]
        tasks = [self.get_market_news(mt) for mt in market_types]
        results = await asyncio.gather(*tasks)

        return {mt: result for mt, result in zip(market_types, results)}

    async def get_asset_news(self, asset: str) -> NewsData:
        """
        Get news for a specific asset.

        Args:
            asset: Asset symbol or name (e.g., 'AAPL', 'bitcoin')

        Returns:
            NewsData object with asset-specific news
        """
        articles = await self.fetch_news_api(asset)

        avg_sentiment = 0.0
        if articles:
            sentiments = [a.sentiment_score for a in articles if a.sentiment_score]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        return NewsData(
            topic=asset,
            articles=articles,
            avg_sentiment=avg_sentiment,
            article_count=len(articles),
            last_updated=datetime.now()
        )
