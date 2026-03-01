"""Async event bus using asyncio.Queue for pub/sub architecture."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List
import time


class EventType(Enum):
    XAU_PRICE = auto()
    XAUT_PRICE = auto()
    MACRO_UPDATE = auto()
    NEWS_ITEM = auto()
    GEO_EVENT = auto()
    SOCIAL_ITEM = auto()
    POLY_UPDATE = auto()
    ANALYSIS_COMPLETE = auto()


@dataclass
class Event:
    type: EventType
    data: Any
    timestamp: float = field(default_factory=time.time)
    source: str = ""


class EventBus:
    """Simple pub/sub event bus backed by asyncio.Queue per subscriber."""

    def __init__(self):
        self._subscribers: Dict[EventType, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, event_type: EventType) -> asyncio.Queue:
        """Subscribe to an event type. Returns a queue that receives events."""
        async with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            q: asyncio.Queue = asyncio.Queue(maxsize=100)
            self._subscribers[event_type].append(q)
            return q

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers of that type."""
        queues = self._subscribers.get(event.type, [])
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest item and insert new one
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    async def publish_many(self, events: List[Event]) -> None:
        """Publish multiple events."""
        for event in events:
            await self.publish(event)


# Global singleton
_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
