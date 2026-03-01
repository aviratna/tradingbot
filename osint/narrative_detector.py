"""Keyword-frequency narrative classifier for XAUUSD macro narratives.

Detects 8 macro narratives from ingested text using keyword matching:
  USD_DOMINANCE, FED_HAWKISH, FED_DOVISH, SAFE_HAVEN_BID,
  INFLATION_FEAR, CENTRAL_BANK_BUYING, RISK_ON, RECESSION_FEAR

- Rolling 200-item deque (most recent text wins via recency weighting)
- Confidence 0..1 per narrative (keyword hit density)
- hawkish_index property (0-100) for FED_HAWKISH vs FED_DOVISH balance
- dominant_narrative: highest confidence NarrativeSignal or None
- Sub-2ms compute time (pure Python, no external deps)
"""

import time
import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Narrative keyword maps
# Each entry: narrative_name → (keyword_list, gold_impact, description)
# gold_impact: "bullish" | "bearish" | "neutral"
# ---------------------------------------------------------------------------

_NARRATIVE_SPECS: Dict[str, dict] = {
    "USD_DOMINANCE": {
        "keywords": [
            "dollar", "usd", "dxy", "dollar index", "dollar strength", "greenback",
            "dollar rally", "dollar surge", "strong dollar", "dollar bull",
            "king dollar", "dollar demand", "reserve currency",
        ],
        "gold_impact": "bearish",
        "description": "Strong US dollar pressures gold prices",
    },
    "FED_HAWKISH": {
        "keywords": [
            "rate hike", "hawkish", "tighten", "tightening", "fed hike",
            "higher rates", "rate increase", "hike", "fed raise",
            "monetary tightening", "restrictive policy", "quantitative tightening",
            "qt", "fed funds rate", "50 bps", "75 bps", "aggressive fed",
        ],
        "gold_impact": "bearish",
        "description": "Fed hawkishness strengthens USD and raises opportunity cost of holding gold",
    },
    "FED_DOVISH": {
        "keywords": [
            "rate cut", "dovish", "pivot", "fed pivot", "easing", "cut rates",
            "rate reduction", "accommodation", "stimulus", "qe",
            "quantitative easing", "pause", "fed pause", "hold rates",
            "lower rates", "rate relief", "soft landing",
        ],
        "gold_impact": "bullish",
        "description": "Fed dovishness weakens USD and supports gold as store of value",
    },
    "SAFE_HAVEN_BID": {
        "keywords": [
            "safe haven", "flight to safety", "risk off", "geopolit",
            "war", "conflict", "tension", "crisis", "sanctions",
            "escalation", "attack", "military", "nato", "ukraine",
            "middle east", "israel", "iran", "north korea",
            "terror", "threat", "nuclear",
        ],
        "gold_impact": "bullish",
        "description": "Geopolitical risk drives safe-haven demand for gold",
    },
    "INFLATION_FEAR": {
        "keywords": [
            "inflation", "cpi", "pce", "price rise", "cost of living",
            "hyperinflation", "stagflation", "purchasing power", "real yield",
            "negative real", "inflationary", "price pressure", "wage inflation",
            "core inflation", "ppi", "supply shock", "energy prices",
        ],
        "gold_impact": "bullish",
        "description": "Inflation fears boost gold as inflation hedge",
    },
    "CENTRAL_BANK_BUYING": {
        "keywords": [
            "central bank", "central banks", "reserve", "gold reserve",
            "pboc", "china buy", "boj", "ecb", "rba", "rbi",
            "official sector", "gold accumulation", "sovereign",
            "imf gold", "gold allocation", "de-dollarization",
            "brics", "gold purchase", "gold demand",
        ],
        "gold_impact": "bullish",
        "description": "Central bank gold purchases provide structural demand support",
    },
    "RISK_ON": {
        "keywords": [
            "risk on", "equity rally", "stock market rally", "bull market",
            "sp500", "nasdaq", "tech rally", "growth stocks",
            "risk appetite", "optimism", "recovery", "expansion",
            "gdp growth", "strong economy", "earnings beat",
            "buy the dip", "btd", "bullish equities",
        ],
        "gold_impact": "bearish",
        "description": "Risk-on environment reduces safe-haven demand for gold",
    },
    "RECESSION_FEAR": {
        "keywords": [
            "recession", "slowdown", "contraction", "gdp decline",
            "growth scare", "hard landing", "layoffs", "unemployment",
            "job losses", "default", "bankruptcy", "credit crunch",
            "yield curve inversion", "inverted yield", "banking crisis",
            "financial crisis", "debt crisis", "contagion",
        ],
        "gold_impact": "bullish",
        "description": "Recession fears drive safe-haven demand and dovish Fed expectations",
    },
}


@dataclass
class NarrativeSignal:
    name: str               # e.g. "FED_HAWKISH"
    confidence: float       # 0..1 — keyword hit density (recency-weighted)
    gold_impact: str        # "bullish" | "bearish" | "neutral"
    description: str        # human-readable explanation
    evidence_count: int     # raw keyword hits in current window
    timestamp: float = field(default_factory=time.time)


@dataclass
class _TextEntry:
    """Internal item stored in the deque."""
    text: str
    timestamp: float = field(default_factory=time.time)


class NarrativeDetector:
    """
    Maintains a rolling deque of text items and computes narrative signals
    via keyword frequency analysis with recency-exponential weighting.

    Compute time: <2ms on 200 items (pure Python, no regex compilation needed).
    Thread-safe: designed for single-threaded async access.
    """

    MAX_ITEMS = 200
    HALF_LIFE_SECONDS = 3600          # 1-hour half-life (same as SentimentEngine)
    MIN_CONFIDENCE_DISPLAY = 0.05     # suppress narratives below this threshold

    def __init__(self):
        self._items: deque = deque(maxlen=self.MAX_ITEMS)
        # Pre-build lowercased keyword sets for fast lookup
        self._specs = {
            name: {
                "keywords": [kw.lower() for kw in spec["keywords"]],
                "gold_impact": spec["gold_impact"],
                "description": spec["description"],
            }
            for name, spec in _NARRATIVE_SPECS.items()
        }

    def add_text(self, text: str) -> None:
        """
        Add a text item to the rolling window. Fast path: skips empty strings.
        """
        if not text or not text.strip():
            return
        self._items.append(_TextEntry(text=text[:800].lower()))

    def detect(self) -> List[NarrativeSignal]:
        """
        Compute narrative confidence scores with exponential recency weighting.
        Returns list sorted by confidence descending.
        Only includes narratives with confidence > MIN_CONFIDENCE_DISPLAY.
        """
        if not self._items:
            return []

        now = time.time()
        decay_k = math.log(2) / self.HALF_LIFE_SECONDS

        # Accumulate weighted keyword hits per narrative
        hit_weights: Dict[str, float] = {name: 0.0 for name in self._specs}
        total_weight = 0.0

        for entry in self._items:
            age = now - entry.timestamp
            weight = math.exp(-decay_k * age)
            total_weight += weight

            for name, spec in self._specs.items():
                for kw in spec["keywords"]:
                    if kw in entry.text:
                        hit_weights[name] += weight
                        break   # count each item once per narrative (prevents keyword spam)

        if total_weight == 0:
            total_weight = 1.0

        results: List[NarrativeSignal] = []
        for name, spec in self._specs.items():
            raw_conf = hit_weights[name] / total_weight
            # Scale so 20% item hit rate → 1.0 confidence
            confidence = min(1.0, raw_conf * 5.0)

            if confidence < self.MIN_CONFIDENCE_DISPLAY:
                continue

            # Approximate evidence_count as items in last 6 hours containing keyword
            cutoff = now - 21600
            evidence_count = sum(
                1 for e in self._items
                if e.timestamp >= cutoff and any(kw in e.text for kw in spec["keywords"])
            )

            results.append(NarrativeSignal(
                name=name,
                confidence=round(confidence, 4),
                gold_impact=spec["gold_impact"],
                description=spec["description"],
                evidence_count=evidence_count,
            ))

        results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    @property
    def hawkish_index(self) -> float:
        """
        Returns 0..100 where:
          0   = purely dovish
          50  = neutral
          100 = purely hawkish
        Computed from FED_HAWKISH vs FED_DOVISH confidence balance.
        """
        narratives = self.detect()
        hawkish_conf = next((n.confidence for n in narratives if n.name == "FED_HAWKISH"), 0.0)
        dovish_conf = next((n.confidence for n in narratives if n.name == "FED_DOVISH"), 0.0)
        total = hawkish_conf + dovish_conf
        if total == 0:
            return 50.0
        return round(50.0 + (hawkish_conf - dovish_conf) / total * 50.0, 2)

    @property
    def dominant_narrative(self) -> Optional[NarrativeSignal]:
        """
        Returns the highest-confidence narrative, or None if no narratives detected.
        """
        narratives = self.detect()
        return narratives[0] if narratives else None

    def clear(self) -> None:
        """Clear all stored items."""
        self._items.clear()
