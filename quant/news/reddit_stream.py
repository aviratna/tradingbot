"""Reddit social sentiment stream using public JSON API (no auth required)."""

import asyncio
import sys
from pathlib import Path
import time
import hashlib
import requests
from dataclasses import dataclass, field
from typing import List, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.event_bus import Event, EventType, get_event_bus
from quant.core.config import get_config
from quant.core.logger import get_logger

logger = get_logger(__name__)

REDDIT_BASE = "https://www.reddit.com/r/{}/new.json"
REDDIT_HEADERS = {
    "User-Agent": "QuantBot/1.0 (gold signal engine; educational purposes)",
    "Accept": "application/json",
}


@dataclass
class SocialItem:
    title: str
    text: str
    subreddit: str
    score: int          # Reddit upvotes
    url: str
    sentiment_score: float
    sentiment_label: str
    relevance_score: float
    timestamp: float = field(default_factory=time.time)
    item_id: str = ""


class RedditStream:
    """Polls Reddit subreddits for gold/macro sentiment."""

    def __init__(self):
        self.config = get_config()
        self.bus = get_event_bus()
        self._seen_ids: Set[str] = set()

    def _score_relevance(self, text: str) -> float:
        """Score relevance to gold/metals/macro."""
        text_lower = text.lower()
        score = 0.0
        for kw in self.config.gold_keywords:
            if kw in text_lower:
                score += 0.15
        return min(score, 1.0)

    def _analyze_sentiment(self, text: str) -> tuple:
        """VADER sentiment."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            compound = analyzer.polarity_scores(text)["compound"]
            label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
            return compound, label
        except Exception:
            return 0.0, "neutral"

    def _fetch_subreddit(self, subreddit: str) -> List[SocialItem]:
        """Fetch latest posts from a subreddit."""
        items = []
        try:
            url = REDDIT_BASE.format(subreddit)
            resp = requests.get(url, headers=REDDIT_HEADERS, timeout=10)
            if resp.status_code == 429:
                logger.warning("reddit_rate_limited", subreddit=subreddit)
                return items
            resp.raise_for_status()
            data = resp.json()
            posts = data.get("data", {}).get("children", [])

            for post in posts[:15]:
                p = post.get("data", {})
                title = p.get("title", "")
                selftext = p.get("selftext", "")[:300]
                score = p.get("score", 0)
                post_id = p.get("id", "")
                permalink = "https://reddit.com" + p.get("permalink", "")

                item_id = hashlib.md5(f"{post_id}{subreddit}".encode()).hexdigest()
                if item_id in self._seen_ids:
                    continue

                text = f"{title} {selftext}"
                relevance = self._score_relevance(text)
                if relevance < 0.08:
                    continue

                sentiment, label = self._analyze_sentiment(text)

                item = SocialItem(
                    title=title[:200],
                    text=selftext,
                    subreddit=subreddit,
                    score=score,
                    url=permalink,
                    sentiment_score=sentiment,
                    sentiment_label=label,
                    relevance_score=relevance,
                    item_id=item_id,
                )
                items.append(item)
                self._seen_ids.add(item_id)
                if len(self._seen_ids) > 2000:
                    self._seen_ids = set(list(self._seen_ids)[-1000:])

        except Exception as e:
            logger.warning("reddit_fetch_failed", subreddit=subreddit, error=str(e))

        return items

    def _fetch_all(self) -> List[SocialItem]:
        """Fetch from all configured subreddits."""
        all_items = []
        for subreddit in self.config.reddit_subreddits:
            items = self._fetch_subreddit(subreddit)
            all_items.extend(items)
            time.sleep(1)  # Be polite to Reddit's rate limits
        return all_items

    async def run(self) -> None:
        """Main stream loop."""
        logger.info("reddit_stream_started")
        while True:
            try:
                items = await asyncio.get_event_loop().run_in_executor(None, self._fetch_all)
                for item in items:
                    event = Event(type=EventType.SOCIAL_ITEM, data=item, source="reddit_stream")
                    await self.bus.publish(event)
                if items:
                    logger.debug("reddit_items_published", count=len(items))
            except asyncio.CancelledError:
                logger.info("reddit_stream_stopped")
                break
            except Exception as e:
                logger.error("reddit_stream_error", error=str(e))

            await asyncio.sleep(self.config.reddit_poll_interval)
