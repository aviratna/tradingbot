"""Polymarket prediction market data fetcher for geopolitical event intelligence."""
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import httpx


GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# Tags/categories relevant to precious metals & macro
RELEVANT_TAGS = [
    "politics", "finance", "economy", "geopolitics", "commodities",
    "federal-reserve", "inflation", "sanctions", "war", "middle-east",
    "russia", "ukraine", "china", "tariffs", "gold", "crypto"
]

# Keywords that are material to XAU/XAG prices
METALS_KEYWORDS = [
    "gold", "silver", "inflation", "fed", "federal reserve", "interest rate",
    "rate cut", "rate hike", "recession", "gdp", "cpi", "iran", "war",
    "sanctions", "dollar", "dxy", "treasury", "yield", "tariff", "china",
    "ukraine", "russia", "oil", "energy", "safe haven", "risk off",
    "geopolit", "conflict", "missile", "nuclear"
]


@dataclass
class PolymarketEvent:
    """A Polymarket prediction market event."""
    id: str
    title: str
    description: str
    category: str
    volume: float
    liquidity: float
    markets: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_active: bool = True
    # Derived fields
    metals_relevance: float = 0.0
    relevance_reason: str = ""
    # Top market prices
    yes_price: float = 0.5
    no_price: float = 0.5
    question: str = ""


@dataclass
class PolymarketFeed:
    """Aggregated Polymarket feed for metals intelligence."""
    trending_events: List[PolymarketEvent]
    metals_relevant: List[PolymarketEvent]
    geopolitical: List[PolymarketEvent]
    total_volume_24h: float
    timestamp: datetime


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _compute_metals_relevance(event: PolymarketEvent) -> tuple[float, str]:
    """Score 0-1 how relevant this event is to XAU/XAG prices."""
    text = (event.title + " " + event.description + " " + event.category).lower()
    score = 0.0
    reasons = []

    keyword_weights = {
        "gold": 1.0, "silver": 1.0, "xau": 1.0, "xag": 1.0,
        "federal reserve": 0.9, "fed ": 0.8, "interest rate": 0.85,
        "rate cut": 0.85, "rate hike": 0.85, "inflation": 0.8,
        "cpi": 0.75, "recession": 0.7, "gdp": 0.6,
        "iran": 0.75, "nuclear": 0.75, "war": 0.65,
        "sanctions": 0.65, "ukraine": 0.6, "russia": 0.6,
        "china": 0.55, "dollar": 0.6, "dxy": 0.7,
        "treasury": 0.65, "yield": 0.6, "tariff": 0.6,
        "oil": 0.5, "energy": 0.45, "safe haven": 0.8,
        "geopolit": 0.6, "conflict": 0.55, "missile": 0.65,
        "middle east": 0.7, "opec": 0.5
    }

    matched = []
    for kw, weight in keyword_weights.items():
        if kw in text:
            score = max(score, weight)
            matched.append(kw.strip())

    if matched:
        reasons = f"Relevant to metals via: {', '.join(matched[:3])}"
    else:
        reasons = ""

    return min(score, 1.0), reasons


class PolymarketFetcher:
    """Fetches prediction market data from Polymarket public APIs."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_ts: Dict[str, datetime] = {}
        self._cache_ttl = 120  # 2 minutes

    def _cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        age = (datetime.now() - self._cache_ts.get(key, datetime.min)).total_seconds()
        return age < self._cache_ttl

    def _set_cache(self, key: str, val: Any) -> None:
        self._cache[key] = val
        self._cache_ts[key] = datetime.now()

    async def get_trending_events(self, limit: int = 30) -> List[PolymarketEvent]:
        """Fetch top trending Polymarket events by volume."""
        cache_key = f"trending_{limit}"
        if self._cache_valid(cache_key):
            return self._cache[cache_key]

        events: List[PolymarketEvent] = []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{GAMMA_API}/events",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": limit,
                        "_sort": "volume",
                        "_order": "DESC"
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data:
                        event = self._parse_event(item)
                        events.append(event)
        except Exception as e:
            print(f"[Polymarket] Error fetching trending events: {e}")

        self._set_cache(cache_key, events)
        return events

    async def get_metals_relevant_events(self) -> List[PolymarketEvent]:
        """Fetch events most relevant to precious metals trading."""
        cache_key = "metals_relevant"
        if self._cache_valid(cache_key):
            return self._cache[cache_key]

        # Get a broad set of active events
        all_events = await self.get_trending_events(limit=50)

        # Score and filter by metals relevance
        relevant = []
        for event in all_events:
            score, reason = _compute_metals_relevance(event)
            event.metals_relevance = score
            event.relevance_reason = reason
            if score >= 0.45:
                relevant.append(event)

        # Sort by relevance * volume
        relevant.sort(key=lambda e: e.metals_relevance * (e.volume + 1), reverse=True)

        self._set_cache(cache_key, relevant[:15])
        return relevant[:15]

    async def get_geopolitical_events(self) -> List[PolymarketEvent]:
        """Fetch geopolitical prediction markets."""
        cache_key = "geopolitical"
        if self._cache_valid(cache_key):
            return self._cache[cache_key]

        events: List[PolymarketEvent] = []
        geo_keywords = ["war", "iran", "russia", "ukraine", "china", "nuclear",
                        "sanctions", "conflict", "military", "tariff", "nato"]

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{GAMMA_API}/events",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": 50,
                        "_sort": "volume",
                        "_order": "DESC"
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data:
                        title = (item.get("title", "") + " " + item.get("description", "")).lower()
                        if any(kw in title for kw in geo_keywords):
                            events.append(self._parse_event(item))
        except Exception as e:
            print(f"[Polymarket] Error fetching geopolitical events: {e}")

        events.sort(key=lambda e: e.volume, reverse=True)
        self._set_cache(cache_key, events[:10])
        return events[:10]

    async def get_market_feed(self) -> PolymarketFeed:
        """Get aggregated Polymarket feed."""
        trending, metals_rel, geo = await asyncio.gather(
            self.get_trending_events(20),
            self.get_metals_relevant_events(),
            self.get_geopolitical_events()
        )

        total_volume = sum(e.volume for e in trending)

        return PolymarketFeed(
            trending_events=trending,
            metals_relevant=metals_rel,
            geopolitical=geo,
            total_volume_24h=total_volume,
            timestamp=datetime.now()
        )

    def _parse_event(self, item: Dict[str, Any]) -> PolymarketEvent:
        """Parse raw API response into PolymarketEvent."""
        markets_raw = item.get("markets", [])

        # Extract best market prices (first active market)
        yes_price = 0.5
        no_price = 0.5
        question = ""
        if markets_raw:
            m = markets_raw[0]
            outcomes = m.get("outcomes", "[]")
            if isinstance(outcomes, str):
                import json
                try:
                    outcomes = json.loads(outcomes)
                except Exception:
                    outcomes = []
            out_prices = m.get("outcomePrices", "[]")
            if isinstance(out_prices, str):
                import json
                try:
                    out_prices = json.loads(out_prices)
                except Exception:
                    out_prices = []

            if len(out_prices) >= 2:
                try:
                    yes_price = float(out_prices[0])
                    no_price = float(out_prices[1])
                except Exception:
                    pass
            question = m.get("question", item.get("title", ""))

        tags = []
        for t in item.get("tags", []):
            if isinstance(t, dict):
                tags.append(t.get("slug", t.get("label", "")))
            elif isinstance(t, str):
                tags.append(t)

        return PolymarketEvent(
            id=str(item.get("id", "")),
            title=item.get("title", ""),
            description=item.get("description", "")[:300] if item.get("description") else "",
            category=item.get("category", ""),
            volume=float(item.get("volume", 0) or 0),
            liquidity=float(item.get("liquidity", 0) or 0),
            markets=markets_raw,
            tags=tags,
            start_date=_parse_date(item.get("startDate")),
            end_date=_parse_date(item.get("endDate")),
            is_active=bool(item.get("active", True)),
            yes_price=yes_price,
            no_price=no_price,
            question=question
        )

    def get_price_signal(self, event: PolymarketEvent) -> Dict[str, Any]:
        """Interpret the market price as a directional signal for metals."""
        yes = event.yes_price
        title = event.title.lower()

        # High probability of risk-off event = bullish for gold
        bullish_patterns = ["war", "conflict", "nuclear", "crisis", "sanction",
                            "recession", "default", "collapse", "attack", "strike iran"]
        bearish_patterns = ["rate hike", "rate increase", "fed hike", "growth",
                            "economy expand", "strong dollar"]

        metals_bias = "neutral"
        bias_strength = 0.0

        for p in bullish_patterns:
            if p in title:
                # High yes probability on a risk event = bullish metals
                bias_strength = yes
                metals_bias = "bullish" if yes > 0.5 else "bearish"
                break

        for p in bearish_patterns:
            if p in title:
                bias_strength = yes
                metals_bias = "bearish" if yes > 0.5 else "bullish"
                break

        return {
            "metals_bias": metals_bias,
            "bias_strength": round(bias_strength, 3),
            "yes_probability": round(yes * 100, 1),
            "no_probability": round((1 - yes) * 100, 1)
        }
