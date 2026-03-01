"""X (Twitter) OSINT client for XAUUSD gold sentiment.

Uses X API v2 with Bearer Token authentication.
Returns empty list + is_available=False if TWITTER_BEARER_TOKEN is not set.
Never raises — all failures are swallowed gracefully.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import List

import httpx

logger = logging.getLogger(__name__)

GOLD_SEARCH_QUERY = (
    "(gold OR XAUUSD OR #gold OR bullion OR #inflation OR \"safe haven\" "
    "OR \"rate hike\" OR \"Fed pivot\" OR \"gold price\") lang:en -is:retweet"
)


@dataclass
class Tweet:
    text: str
    author: str
    created_at: float    # unix timestamp
    like_count: int
    retweet_count: int
    reply_count: int
    url: str
    sentiment_score: float   # VADER compound -1..1
    engagement_score: float  # normalized 0..1


def _compute_vader_sentiment(text: str) -> float:
    """Return VADER compound score. Returns 0.0 on failure."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)["compound"]
    except Exception:
        return 0.0


def _normalize_tweet_engagement(tweets: list) -> list:
    """Normalize engagement scores 0..1 across batch."""
    if not tweets:
        return []
    scores = [(t["like_count"] + t["retweet_count"] * 2 + t["reply_count"]) for t in tweets]
    max_s = max(scores) if scores else 1
    if max_s == 0:
        max_s = 1
    return [s / max_s for s in scores]


class TwitterClient:
    """
    Fetches recent gold-related tweets via X API v2.
    • If TWITTER_BEARER_TOKEN is set → queries search/recent endpoint
    • Otherwise → is_available=False, returns [] immediately
    • Rate-limit safe: 3 retries with exponential backoff
    • Never raises.
    """

    _SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
    _MAX_RESULTS = 25
    _TIMEOUT = 10.0

    def __init__(self):
        self._token = os.getenv("TWITTER_BEARER_TOKEN", "").strip()
        self._available = bool(self._token)
        if self._available:
            logger.info("twitter_client_initialized_with_token")
        else:
            logger.info("twitter_client_no_token_stub_mode")

    async def fetch_recent_tweets(self, limit: int = 25) -> List[Tweet]:
        """
        Fetch recent gold-related tweets. Returns [] if unavailable.
        Async — safe to await directly from OsintAggregator._cycle().
        """
        if not self._available:
            return []
        try:
            return await self._fetch(min(limit, 100))
        except Exception as e:
            logger.warning(f"twitter_fetch_failed: {e}")
            return []

    async def _fetch(self, limit: int) -> List[Tweet]:
        """Actual X API v2 call with retry logic."""
        params = {
            "query": GOLD_SEARCH_QUERY,
            "max_results": max(10, min(limit, 100)),
            "tweet.fields": "created_at,public_metrics,author_id",
            "expansions": "author_id",
            "user.fields": "username",
        }
        headers = {"Authorization": f"Bearer {self._token}"}

        last_err = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self._TIMEOUT) as client:
                    resp = await client.get(self._SEARCH_URL, params=params, headers=headers)

                if resp.status_code == 429:
                    # Rate limited — wait and retry
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"twitter_rate_limited, waiting {wait}s")
                    import asyncio
                    await asyncio.sleep(wait)
                    continue

                if resp.status_code != 200:
                    logger.warning(f"twitter_api_error: status={resp.status_code}")
                    return []

                data = resp.json()
                return self._parse_response(data, limit)

            except Exception as e:
                last_err = e
                import asyncio
                await asyncio.sleep(2 ** attempt)

        logger.warning(f"twitter_fetch_all_retries_failed: {last_err}")
        return []

    def _parse_response(self, data: dict, limit: int) -> List[Tweet]:
        """Parse X API v2 response into Tweet list."""
        tweets_raw = data.get("data", [])
        if not tweets_raw:
            return []

        # Build author map from includes
        users = {u["id"]: u.get("username", "unknown")
                 for u in data.get("includes", {}).get("users", [])}

        raw_list = []
        for t in tweets_raw[:limit]:
            metrics = t.get("public_metrics", {})
            author_id = t.get("author_id", "")
            created_at_str = t.get("created_at", "")
            try:
                from datetime import datetime, timezone
                ts = datetime.fromisoformat(
                    created_at_str.replace("Z", "+00:00")
                ).timestamp()
            except Exception:
                ts = time.time()

            raw_list.append({
                "text": t.get("text", ""),
                "author": users.get(author_id, "unknown"),
                "created_at": ts,
                "like_count": metrics.get("like_count", 0),
                "retweet_count": metrics.get("retweet_count", 0),
                "reply_count": metrics.get("reply_count", 0),
                "url": f"https://twitter.com/i/web/status/{t.get('id', '')}",
            })

        eng_norms = _normalize_tweet_engagement(raw_list)
        result: List[Tweet] = []
        for i, t in enumerate(raw_list):
            sentiment = _compute_vader_sentiment(t["text"])
            result.append(Tweet(
                text=t["text"][:280],
                author=t["author"],
                created_at=t["created_at"],
                like_count=t["like_count"],
                retweet_count=t["retweet_count"],
                reply_count=t["reply_count"],
                url=t["url"],
                sentiment_score=round(sentiment, 4),
                engagement_score=round(eng_norms[i], 4) if i < len(eng_norms) else 0.0,
            ))

        result.sort(key=lambda x: x.engagement_score, reverse=True)
        return result

    @property
    def is_available(self) -> bool:
        """True only if TWITTER_BEARER_TOKEN is set."""
        return self._available
