"""
Real-Time Quant Signal Engine — Entry Point
============================================
Run with:  python -m quant.main

Streams all market data, news, and Polymarket feeds asynchronously.
Runs analysis pipeline on each XAU price update.
Renders Rich terminal dashboard with 8 panels.
Exports JSON snapshots and CSV log every 60 seconds.
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple
import asyncio

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.core.event_bus import EventBus, EventType, Event, get_event_bus
from quant.core.stream_manager import StreamManager, setup_signal_handlers
from quant.core.config import get_config
from quant.core.logger import setup_logging, get_logger

from quant.market.xau_stream import XAUStream, XAUData
from quant.market.xaut_stream import XAUTStream, XAUTData
from quant.market.macro_stream import MacroStream, MacroData

from quant.news.finance_stream import FinanceNewsStream
from quant.news.geopolitics_stream import GeopoliticsStream, GeoEvent
from quant.news.reddit_stream import RedditStream

from quant.polymarket.polymarket_stream import PolymarketStream, PolyUpdate

from quant.analysis.technicals import TechnicalSnapshot, compute_technicals
from quant.analysis.correlations import CorrelationSnapshot, CorrelationAnalyzer
from quant.analysis.macro_model import MacroSnapshot, compute_macro
from quant.analysis.sentiment_model import SentimentSnapshot, SentimentModel
from quant.analysis.regime_detector import RegimeSnapshot, RegimeDetector
from quant.analysis.signal_scoring import SignalScore, score_signal
from quant.analysis.forecaster import ForecastSnapshot, compute_forecast

from quant.trade.risk_model import RiskModel, RiskSnapshot
from quant.trade.trade_suggestion import TradeSuggestion, generate_suggestion

from quant.dashboard.live_console import LiveConsole
from quant.dashboard.json_export import JSONExporter

setup_logging("INFO")
logger = get_logger(__name__)


@dataclass
class QuantState:
    """Shared mutable state for all analysis results."""
    # Market data
    xau_data: Optional[XAUData] = None
    xaut_data: Optional[XAUTData] = None
    macro_data: Optional[MacroData] = None

    # Analysis snapshots
    tech_snap: Optional[TechnicalSnapshot] = None
    corr_snap: Optional[CorrelationSnapshot] = None
    macro_snap: Optional[MacroSnapshot] = None
    sentiment_snap: Optional[SentimentSnapshot] = None
    regime_snap: Optional[RegimeSnapshot] = None
    signal_score: Optional[SignalScore] = None
    risk_snap: Optional[RiskSnapshot] = None
    forecast_snap: Optional[ForecastSnapshot] = None
    trade_suggestion: Optional[TradeSuggestion] = None

    # Polymarket
    poly_data: Optional[PolyUpdate] = None

    # Event log: [(timestamp, message, color)]
    recent_events: List[Tuple[float, str, str]] = field(default_factory=list)
    MAX_EVENTS: int = 50

    # Lock for concurrent updates
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def add_event(self, message: str, color: str = "white") -> None:
        """Thread-safe event log append."""
        self.recent_events.append((time.time(), message, color))
        if len(self.recent_events) > self.MAX_EVENTS:
            self.recent_events = self.recent_events[-self.MAX_EVENTS:]


class AnalysisOrchestrator:
    """
    Subscribes to the event bus and triggers the analysis pipeline
    on every XAU price update.
    """

    def __init__(self, state: QuantState, bus: EventBus):
        self._state = state
        self._bus = bus
        self._corr_analyzer = CorrelationAnalyzer()
        self._sentiment_model = SentimentModel()
        self._regime_detector = RegimeDetector()
        self._risk_model = RiskModel()
        self._analysis_lock = asyncio.Lock()
        self._last_daily_update: float = 0

    async def run_event_consumer(self) -> None:
        """Subscribe to all event types and dispatch to handlers."""
        queues = {
            EventType.XAU_PRICE: await self._bus.subscribe(EventType.XAU_PRICE),
            EventType.XAUT_PRICE: await self._bus.subscribe(EventType.XAUT_PRICE),
            EventType.MACRO_UPDATE: await self._bus.subscribe(EventType.MACRO_UPDATE),
            EventType.NEWS_ITEM: await self._bus.subscribe(EventType.NEWS_ITEM),
            EventType.GEO_EVENT: await self._bus.subscribe(EventType.GEO_EVENT),
            EventType.SOCIAL_ITEM: await self._bus.subscribe(EventType.SOCIAL_ITEM),
            EventType.POLY_UPDATE: await self._bus.subscribe(EventType.POLY_UPDATE),
        }

        logger.info("orchestrator_subscribed_to_all_events")

        async def _consume(event_type: EventType, queue: asyncio.Queue):
            while True:
                try:
                    event: Event = await queue.get()
                    await self._dispatch(event)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("event_dispatch_error", type=event_type.name, error=str(e))

        # Run all consumers concurrently
        tasks = [asyncio.create_task(_consume(et, q)) for et, q in queues.items()]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()

    async def _dispatch(self, event: Event) -> None:
        """Route events to the appropriate handler."""
        et = event.type
        data = event.data

        if et == EventType.XAU_PRICE:
            self._state.xau_data = data
            self._state.add_event(
                f"XAU/USD ${data.price:,.2f} ({data.change_pct:+.2f}%)",
                "green" if data.change_pct >= 0 else "red",
            )
            # Update correlation tracker with price point every hour (approx daily)
            now = time.time()
            if now - self._last_daily_update > 3600:
                self._corr_analyzer.update_xau(data.price)
                self._last_daily_update = now
            # Trigger full analysis pipeline
            await self._run_analysis_pipeline(data)

        elif et == EventType.XAUT_PRICE:
            self._state.xaut_data = data
            self._corr_analyzer.update_xaut(data.price)
            if abs(data.premium_discount_pct) > 0.5:
                self._state.add_event(
                    f"XAUT ${data.price:,.2f} ({data.premium_discount_pct:+.2f}% vs spot)",
                    "magenta",
                )

        elif et == EventType.MACRO_UPDATE:
            self._state.macro_data = data
            # Update correlation model with macro prices
            self._corr_analyzer.update_macro(
                dxy=data.dxy.price if data.dxy else None,
                spy=data.spy.price if data.spy else None,
                vix=data.vix.price if data.vix else None,
                uso=data.uso.price if data.uso else None,
            )
            vix = data.vix.price if data.vix else None
            if vix and vix > 25:
                self._state.add_event(f"⚠️ VIX elevated at {vix:.1f}", "yellow")

        elif et == EventType.NEWS_ITEM:
            self._sentiment_model.add_news(data)
            self._state.add_event(
                f"📰 {data.source}: {data.title[:60]}",
                "green" if data.sentiment_score > 0.1 else "red" if data.sentiment_score < -0.1 else "dim",
            )

        elif et == EventType.GEO_EVENT:
            self._sentiment_model.add_geo(data)
            color = "bright_red" if data.severity_label == "critical" else "orange1" if data.severity_label == "high" else "yellow"
            self._state.add_event(
                f"🌍 [{data.severity_label.upper()}] {data.title[:55]}",
                color,
            )

        elif et == EventType.SOCIAL_ITEM:
            self._sentiment_model.add_reddit(data)

        elif et == EventType.POLY_UPDATE:
            self._state.poly_data = data
            self._state.add_event(
                f"🎯 Polymarket: {data.overall_bias.upper()} bias | risk-off={data.risk_off_index:.2f}",
                "cyan",
            )

    async def _run_analysis_pipeline(self, xau_data) -> None:
        """
        Run full analysis pipeline in executor (CPU-bound).
        Called on every XAU price event.
        """
        async with self._analysis_lock:
            try:
                loop = asyncio.get_event_loop()
                state = self._state

                def _pipeline():
                    bars = xau_data.bars
                    price = xau_data.price

                    # 1. Technicals
                    tech = compute_technicals(bars) if len(bars) >= 30 else None

                    # 2. Correlations
                    corr = self._corr_analyzer.compute()

                    # 3. Macro model
                    macro_snap = compute_macro(state.macro_data) if state.macro_data else None

                    # 4. Sentiment
                    sent_snap = self._sentiment_model.compute()

                    # 5. Regime
                    geo_sev = 0.0
                    if state.recent_events:
                        # Average severity from geo events in last 5 min
                        geo_events = [e for e in state.recent_events
                                      if "[CRITICAL]" in e[1] or "[HIGH]" in e[1]]
                        if geo_events:
                            geo_sev = min(len(geo_events) * 0.2, 1.0)

                    poly_risk = state.poly_data.risk_off_index if state.poly_data else 0.3
                    regime = self._regime_detector.detect(
                        macro_data=state.macro_data,
                        geo_sentiment=geo_sev,
                        poly_risk_off_index=poly_risk,
                    )

                    # 6. Signal scoring
                    sig_score = score_signal(tech, macro_snap, sent_snap, state.poly_data, regime)

                    # 7. Risk model
                    risk = self._risk_model.compute(bars, price) if len(bars) >= 15 else None

                    # 8. Forecaster
                    forecast = compute_forecast(bars, price) if len(bars) >= 30 else None

                    # 9. Trade suggestion
                    trade = generate_suggestion(
                        current_price=price,
                        tech_snap=tech,
                        signal_snap=sig_score,
                        risk_snap=risk,
                        regime_snap=regime,
                        forecast_snap=forecast,
                    )

                    return tech, corr, macro_snap, sent_snap, regime, sig_score, risk, forecast, trade

                results = await loop.run_in_executor(None, _pipeline)
                (
                    state.tech_snap,
                    state.corr_snap,
                    state.macro_snap,
                    state.sentiment_snap,
                    state.regime_snap,
                    state.signal_score,
                    state.risk_snap,
                    state.forecast_snap,
                    state.trade_suggestion,
                ) = results

                # Log significant signal changes
                if state.signal_score:
                    score = state.signal_score.composite
                    if score >= 70:
                        state.add_event(
                            f"🚀 STRONG BUY signal: {score:.0f}/100 | {state.regime_snap.regime.value if state.regime_snap else ''}",
                            "bright_green",
                        )
                    elif score <= 30:
                        state.add_event(
                            f"🔻 STRONG SELL signal: {score:.0f}/100",
                            "bright_red",
                        )

                await self._bus.publish(Event(
                    type=EventType.ANALYSIS_COMPLETE,
                    data=state.signal_score,
                    source="orchestrator",
                ))

            except Exception as e:
                logger.error("analysis_pipeline_error", error=str(e), exc_info=True)


async def main():
    """Main async entry point."""
    config = get_config()
    bus = get_event_bus()
    manager = StreamManager()
    state = QuantState()

    # XAU price reference shared with XAUT stream
    xau_price_ref = [0.0]

    async def _xau_price_tracker():
        """Update shared price reference for XAUT premium calc."""
        q = await bus.subscribe(EventType.XAU_PRICE)
        while True:
            event = await q.get()
            xau_price_ref[0] = event.data.price

    logger.info("quant_engine_starting")
    print("\n" + "="*60)
    print("  🏆 XAU Quant Signal Engine")
    print("  Focused: XAU/USD + XAUT/USDT")
    print("  Streams: Market + News + Reddit + Polymarket")
    print("  Dashboard: Rich terminal (8 panels)")
    print("="*60 + "\n")

    # Initialize all streams
    xau_stream = XAUStream()
    xaut_stream = XAUTStream(xau_price_ref)
    macro_stream = MacroStream()
    finance_stream = FinanceNewsStream()
    geo_stream = GeopoliticsStream()
    reddit_stream = RedditStream()
    poly_stream = PolymarketStream()
    orchestrator = AnalysisOrchestrator(state, bus)
    dashboard = LiveConsole(state)
    exporter = JSONExporter(state)

    await manager.start_all()

    # Register all tasks
    manager.register("xau_stream", xau_stream.run())
    manager.register("xaut_stream", xaut_stream.run())
    manager.register("macro_stream", macro_stream.run())
    manager.register("finance_stream", finance_stream.run())
    manager.register("geo_stream", geo_stream.run())
    manager.register("reddit_stream", reddit_stream.run())
    manager.register("poly_stream", poly_stream.run())
    manager.register("xau_price_tracker", _xau_price_tracker())
    manager.register("orchestrator", orchestrator.run_event_consumer())
    manager.register("dashboard", dashboard.run())
    manager.register("exporter", exporter.run())

    # Set up graceful shutdown on SIGINT/SIGTERM
    loop = asyncio.get_event_loop()
    setup_signal_handlers(manager, loop)

    logger.info("all_streams_registered", count=11)

    try:
        await manager.wait_all()
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt_received")
        await manager.stop_all()
    finally:
        logger.info("quant_engine_stopped")


if __name__ == "__main__":
    asyncio.run(main())
