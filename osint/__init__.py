"""OSINT Intelligence Layer for XAUUSD quant trading system.

Provides:
- Reddit/Twitter social data ingestion (with graceful fallbacks)
- VADER-based sentiment + fear/optimism scoring
- Keyword-frequency narrative detection
- Claude AI macro summarization (3-min cache)
- Gold Risk Index (GRI) composite scoring
- <100ms fast signal pulse
- Trade bias adapter (risk control multipliers)
- Master OsintAggregator coordinator

Usage:
    from osint.osint_aggregator import OsintAggregator
    aggregator = OsintAggregator(state, bus)
    await aggregator.run()
"""

from .osint_aggregator import OsintAggregator, OsintSnapshot

__all__ = ["OsintAggregator", "OsintSnapshot"]
