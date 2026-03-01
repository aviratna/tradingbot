"""AnalysisOrchestrator: subscribes to event bus, runs analysis pipeline on XAU ticks."""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.event_bus import EventBus, EventType, Event
from quant.core.logger import get_logger
from quant.analysis.technicals import compute_technicals
from quant.analysis.correlations import CorrelationAnalyzer
from quant.analysis.macro_model import compute_macro
from quant.analysis.sentiment_model import SentimentModel
from quant.analysis.regime_detector import RegimeDetector
from quant.analysis.signal_scoring import score_signal
from quant.analysis.forecaster import compute_forecast
from quant.trade.risk_model import RiskModel
from quant.trade.trade_suggestion import generate_suggestion

logger = get_logger(__name__)


class AnalysisOrchestrator:
    """
    Subscribes to the event bus and triggers the analysis pipeline
    on every XAU price update. Updates QuantState in-place.
    """

    def __init__(self, state, bus: EventBus):
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
        state = self._state

        if et == EventType.XAU_PRICE:
            state.xau_data = data
            state.add_event(
                f"XAU/USD ${data.price:,.2f} ({data.change_pct:+.2f}%)",
                "green" if data.change_pct >= 0 else "red",
            )
            now = time.time()
            if now - self._last_daily_update > 3600:
                self._corr_analyzer.update_xau(data.price)
                self._last_daily_update = now
            await self._run_analysis_pipeline(data)

        elif et == EventType.XAUT_PRICE:
            state.xaut_data = data
            self._corr_analyzer.update_xaut(data.price)
            if abs(data.premium_discount_pct) > 0.5:
                state.add_event(
                    f"XAUT ${data.price:,.2f} ({data.premium_discount_pct:+.2f}% vs spot)",
                    "magenta",
                )

        elif et == EventType.MACRO_UPDATE:
            state.macro_data = data
            self._corr_analyzer.update_macro(
                dxy=data.dxy.price if data.dxy else None,
                spy=data.spy.price if data.spy else None,
                vix=data.vix.price if data.vix else None,
                uso=data.uso.price if data.uso else None,
            )
            vix = data.vix.price if data.vix else None
            if vix and vix > 25:
                state.add_event(f"VIX elevated at {vix:.1f}", "yellow")

        elif et == EventType.NEWS_ITEM:
            self._sentiment_model.add_news(data)
            state.add_event(
                f"NEWS {data.source}: {data.title[:55]}",
                "green" if data.sentiment_score > 0.1 else "red" if data.sentiment_score < -0.1 else "dim",
            )

        elif et == EventType.GEO_EVENT:
            self._sentiment_model.add_geo(data)
            color = "bright_red" if data.severity_label == "critical" else "orange1" if data.severity_label == "high" else "yellow"
            state.add_event(
                f"[{data.severity_label.upper()}] {data.title[:55]}",
                color,
            )

        elif et == EventType.SOCIAL_ITEM:
            self._sentiment_model.add_reddit(data)

        elif et == EventType.POLY_UPDATE:
            state.poly_data = data
            state.add_event(
                f"Polymarket: {data.overall_bias.upper()} bias | risk-off={data.risk_off_index:.2f}",
                "cyan",
            )

    async def _run_analysis_pipeline(self, xau_data) -> None:
        """Run full analysis pipeline in executor."""
        async with self._analysis_lock:
            try:
                loop = asyncio.get_event_loop()
                state = self._state

                def _pipeline():
                    bars = xau_data.bars
                    price = xau_data.price

                    tech = compute_technicals(bars) if len(bars) >= 30 else None
                    corr = self._corr_analyzer.compute()
                    macro_snap = compute_macro(state.macro_data) if state.macro_data else None
                    sent_snap = self._sentiment_model.compute()

                    geo_events = [e for e in state.recent_events
                                  if "[CRITICAL]" in e[1] or "[HIGH]" in e[1]]
                    geo_sev = min(len(geo_events) * 0.2, 1.0) if geo_events else 0.0
                    poly_risk = state.poly_data.risk_off_index if state.poly_data else 0.3

                    regime = self._regime_detector.detect(
                        macro_data=state.macro_data,
                        geo_sentiment=geo_sev,
                        poly_risk_off_index=poly_risk,
                    )
                    sig_score = score_signal(tech, macro_snap, sent_snap, state.poly_data, regime)
                    risk = self._risk_model.compute(bars, price) if len(bars) >= 15 else None
                    forecast = compute_forecast(bars, price) if len(bars) >= 30 else None
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

                if state.signal_score:
                    score = state.signal_score.composite
                    if score >= 70:
                        state.add_event(
                            f"STRONG BUY signal: {score:.0f}/100 | {state.regime_snap.regime.value if state.regime_snap else ''}",
                            "bright_green",
                        )
                    elif score <= 30:
                        state.add_event(f"STRONG SELL signal: {score:.0f}/100", "bright_red")

                await self._bus.publish(Event(
                    type=EventType.ANALYSIS_COMPLETE,
                    data=state.signal_score,
                    source="orchestrator",
                ))

            except Exception as e:
                logger.error("analysis_pipeline_error", error=str(e), exc_info=True)
