"""Shared QuantState dataclass used by both standalone CLI and FastAPI integration."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class QuantState:
    """Shared mutable state for all analysis results."""
    # Market data (populated by streams)
    xau_data: Optional[object] = None
    xaut_data: Optional[object] = None
    macro_data: Optional[object] = None

    # Analysis snapshots (populated by orchestrator)
    tech_snap: Optional[object] = None
    corr_snap: Optional[object] = None
    macro_snap: Optional[object] = None
    sentiment_snap: Optional[object] = None
    regime_snap: Optional[object] = None
    signal_score: Optional[object] = None
    risk_snap: Optional[object] = None
    forecast_snap: Optional[object] = None
    trade_suggestion: Optional[object] = None

    # Polymarket
    poly_data: Optional[object] = None

    # OSINT intelligence layer (populated by OsintAggregator every 3 min)
    osint_data: Optional[object] = None

    # Event log: [(timestamp, message, color)]
    recent_events: List[Tuple[float, str, str]] = field(default_factory=list)
    MAX_EVENTS: int = 50

    def add_event(self, message: str, color: str = "white") -> None:
        """Append to in-memory event log."""
        self.recent_events.append((time.time(), message, color))
        if len(self.recent_events) > self.MAX_EVENTS:
            self.recent_events = self.recent_events[-self.MAX_EVENTS:]
