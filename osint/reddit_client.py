"""Reddit OSINT client for XAUUSD gold sentiment.

Uses PRAW OAuth if REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET are set.
Falls back to the public Reddit JSON API (no auth required) — same approach
as quant/news/reddit_stream.py. Never raises on failure.
"""

import os
import time
import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

GOLD_KEYWORDS = [
    "gold", "xau", "xauusd", "silver", "precious metal", "bullion",
    "inflation", "fed", "federal reserve", "dollar", "dxy", "yield",
    "treasury", "war", "conflict", "sanctions", "geopolit", "safe haven",
    "rate hike", "rate cut", "recession", "stagflation", "risk off",
]

SUBREDDITS = [
    "investing", "wallstreetbets", "economics", "geopolitics",
    "gold", "Silverbugs", "forex", "personalfinance",
]


@dataclass
class RedditPost:
    title: str
    text: str
    subreddit: str
    score: int              # upvote score
    num_comments: int
    url: str
    sentiment_score: float  # VADER compound -1..1
    engagement_score: float # normalized 0..1
    created_utc: float
    item_id: str = ""
    upvote_ratio: float = 0.0


def _compute_vader_sentiment(text: str) -> float:
    """Return VADER compound score -1..1. Returns 0.0 on any failure."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)["compound"]
    except Exception:
        return 0.0


def _is_gold_relevant(text: str) -> bool:
    """Returns True if text contains any gold-relevant keyword."""
    lower = text.lower()
    return any(kw in lower for kw in GOLD_KEYWORDS)


def _normalize_engagement(posts: List[dict]) -> List[float]:
    """Normalize engagement scores 0..1 across a batch."""
    if not posts:
        return []
    scores = [p.get("score", 0) + p.get("num_comments", 0) * 2 for p in posts]
    max_s = max(scores) if scores else 1
    if max_s == 0:
        max_s = 1
    return [s / max_s for s in scores]


class RedditClient:
    """
    Fetches gold-relevant Reddit posts.
    • If REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET set → uses PRAW OAuth
    • Otherwise → falls back to public Reddit JSON API (no auth)
    • Always returns gracefully; never raises.
    """

    _HEADERS = {"User-Agent": os.getenv("REDDIT_USER_AGENT", "OSINTBot/1.0 (gold quant)")}
    _PUBLIC_BASE = "https://www.reddit.com/r/{sub}/hot.json"
    _TIMEOUT = 8  # seconds

    def __init__(self):
        self._praw_available = False
        self._reddit = None
        client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
        client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
        if client_id and client_secret:
            self._init_praw(client_id, client_secret)

    def _init_praw(self, client_id: str, client_secret: str) -> None:
        """Attempt to initialize PRAW; silently disable if unavailable."""
        try:
            import praw
            self._reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=os.getenv("REDDIT_USER_AGENT", "OSINTBot/1.0"),
            )
            # Test connection (fast head call)
            self._reddit.user.me()   # raises if credentials invalid
            self._praw_available = True
            logger.info("reddit_client_praw_initialized")
        except Exception as e:
            logger.warning(f"reddit_praw_init_failed: {e} — using public JSON fallback")
            self._praw_available = False
            self._reddit = None

    def fetch_top_posts(self, limit: int = 25) -> List[RedditPost]:
        """
        Fetch top gold-relevant Reddit posts. Returns [] on any failure.
        Runs synchronously (call via run_in_executor from async context).
        """
        try:
            if self._praw_available and self._reddit:
                return self._fetch_praw(limit)
            return self._fetch_public_json(limit)
        except Exception as e:
            logger.warning(f"reddit_fetch_failed: {e}")
            return []

    def _fetch_praw(self, limit: int) -> List[RedditPost]:
        """Fetch using PRAW OAuth (richer data including upvote_ratio)."""
        posts: List[RedditPost] = []
        raw_posts = []
        for sub_name in SUBREDDITS:
            try:
                sub = self._reddit.subreddit(sub_name)
                for submission in sub.hot(limit=max(10, limit // len(SUBREDDITS))):
                    combined = f"{submission.title} {submission.selftext or ''}"
                    if _is_gold_relevant(combined):
                        raw_posts.append({
                            "title": submission.title,
                            "text": (submission.selftext or "")[:500],
                            "subreddit": sub_name,
                            "score": submission.score,
                            "num_comments": submission.num_comments,
                            "url": f"https://reddit.com{submission.permalink}",
                            "created_utc": submission.created_utc,
                            "item_id": submission.id,
                            "upvote_ratio": getattr(submission, "upvote_ratio", 0.0),
                        })
            except Exception as e:
                logger.debug(f"reddit_praw_sub_failed: r/{sub_name}: {e}")
                continue

        eng_norms = _normalize_engagement(raw_posts)
        for i, p in enumerate(raw_posts[:limit]):
            sentiment = _compute_vader_sentiment(f"{p['title']} {p['text']}")
            posts.append(RedditPost(
                title=p["title"][:200],
                text=p["text"][:500],
                subreddit=p["subreddit"],
                score=p["score"],
                num_comments=p["num_comments"],
                url=p["url"],
                sentiment_score=round(sentiment, 4),
                engagement_score=round(eng_norms[i], 4) if i < len(eng_norms) else 0.0,
                created_utc=p["created_utc"],
                item_id=p.get("item_id", ""),
                upvote_ratio=p.get("upvote_ratio", 0.0),
            ))

        posts.sort(key=lambda x: x.engagement_score, reverse=True)
        return posts[:limit]

    def _fetch_public_json(self, limit: int) -> List[RedditPost]:
        """
        Fetch via public Reddit JSON API (no auth required).
        Same approach as quant/news/reddit_stream.py.
        """
        raw_posts = []
        per_sub = max(5, limit // len(SUBREDDITS))

        for sub_name in SUBREDDITS:
            try:
                url = self._PUBLIC_BASE.format(sub=sub_name) + f"?limit={per_sub}"
                resp = requests.get(url, headers=self._HEADERS, timeout=self._TIMEOUT)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                children = data.get("data", {}).get("children", [])
                for child in children:
                    p = child.get("data", {})
                    title = p.get("title", "")
                    text = p.get("selftext", "")
                    combined = f"{title} {text}"
                    if not _is_gold_relevant(combined):
                        continue
                    raw_posts.append({
                        "title": title,
                        "text": text[:500],
                        "subreddit": sub_name,
                        "score": p.get("score", 0),
                        "num_comments": p.get("num_comments", 0),
                        "url": f"https://reddit.com{p.get('permalink', '')}",
                        "created_utc": p.get("created_utc", time.time()),
                        "item_id": p.get("id", ""),
                        "upvote_ratio": p.get("upvote_ratio", 0.0),
                    })
            except Exception as e:
                logger.debug(f"reddit_public_json_sub_failed: r/{sub_name}: {e}")
                continue

        # Sort by score + recency
        now = time.time()
        raw_posts.sort(
            key=lambda p: p["score"] * math.exp(-(now - p["created_utc"]) / 86400),
            reverse=True
        )

        eng_norms = _normalize_engagement(raw_posts)
        posts: List[RedditPost] = []
        for i, p in enumerate(raw_posts[:limit]):
            sentiment = _compute_vader_sentiment(f"{p['title']} {p['text']}")
            posts.append(RedditPost(
                title=p["title"][:200],
                text=p["text"][:500],
                subreddit=p["subreddit"],
                score=p["score"],
                num_comments=p["num_comments"],
                url=p["url"],
                sentiment_score=round(sentiment, 4),
                engagement_score=round(eng_norms[i], 4) if i < len(eng_norms) else 0.0,
                created_utc=p["created_utc"],
                item_id=p.get("item_id", ""),
                upvote_ratio=p.get("upvote_ratio", 0.0),
            ))

        return posts

    @property
    def is_available(self) -> bool:
        """Always True — public JSON fallback guarantees some data."""
        return True

    @property
    def uses_praw(self) -> bool:
        """True if PRAW OAuth is active."""
        return self._praw_available
