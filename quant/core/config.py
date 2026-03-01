"""Quant engine configuration."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class QuantConfig:
    # Poll intervals (seconds)
    xau_poll_interval: int = 30
    xaut_poll_interval: int = 30
    macro_poll_interval: int = 60
    news_poll_interval: int = 300   # 5 min
    reddit_poll_interval: int = 300  # 5 min
    polymarket_poll_interval: int = 120  # 2 min
    export_interval: int = 60

    # Analysis windows
    technical_bars: int = 200        # bars for technical analysis
    correlation_window: int = 30     # days for rolling correlation
    sentiment_window: int = 50       # recent items for sentiment
    forecast_bars: int = 100         # bars for ARIMA

    # RSS feeds
    finance_rss_feeds: List[str] = field(default_factory=lambda: [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/reuters/companyNews",
        "http://feeds.marketwatch.com/marketwatch/topstories/",
        "https://finance.yahoo.com/news/rssindex",
    ])

    geopolitics_rss_feeds: List[str] = field(default_factory=lambda: [
        "https://www.aljazeera.com/xml/rss/all.xml",
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    ])

    # Reddit subreddits for gold/macro sentiment
    reddit_subreddits: List[str] = field(default_factory=lambda: [
        "gold", "wallstreetbets", "investing", "economics", "Silverbugs"
    ])

    # Gold-relevant keywords for filtering
    gold_keywords: List[str] = field(default_factory=lambda: [
        "gold", "xau", "silver", "precious metal", "bullion", "inflation",
        "fed", "federal reserve", "dollar", "dxy", "yield", "treasury",
        "war", "conflict", "sanctions", "geopolit", "risk off", "safe haven",
        "crypto", "bitcoin", "tether gold", "xaut", "stagflation"
    ])

    # Signal scoring weights
    tech_weight: float = 0.30
    macro_weight: float = 0.30
    sentiment_weight: float = 0.20
    polymarket_weight: float = 0.20

    # ATR multipliers for trade zones
    entry_atr_mult: float = 0.3
    sl_atr_mult: float = 1.5
    tp1_atr_mult: float = 1.5
    tp2_atr_mult: float = 2.5

    # Regime thresholds
    vix_risk_off: float = 25.0
    vix_liquidity_squeeze: float = 35.0
    yield_hawkish: float = 4.5
    dxy_hawkish_5d_change: float = 1.0
    uso_inflation_5d_change: float = 3.0
    geo_severity_war: float = 0.7
    polymarket_risk_off_index: float = 0.6
    spy_liquidity_5d_change: float = -5.0


# Global singleton
_config: QuantConfig | None = None


def get_config() -> QuantConfig:
    global _config
    if _config is None:
        _config = QuantConfig()
    return _config
