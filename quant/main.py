"""
Real-Time Quant Signal Engine — Standalone CLI Entry Point
==========================================================
Run with:  python -m quant.main

Streams all market data, news, and Polymarket feeds asynchronously.
Runs analysis pipeline on each XAU price update.
Renders Rich terminal dashboard with 8 panels.
Exports JSON snapshots and CSV log every 60 seconds.
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.core.event_bus import EventType, get_event_bus
from quant.core.stream_manager import StreamManager, setup_signal_handlers
from quant.core.logger import setup_logging, get_logger
from quant.state import QuantState
from quant.analysis.orchestrator import AnalysisOrchestrator

from quant.market.xau_stream import XAUStream
from quant.market.xaut_stream import XAUTStream
from quant.market.macro_stream import MacroStream
from quant.news.finance_stream import FinanceNewsStream
from quant.news.geopolitics_stream import GeopoliticsStream
from quant.news.reddit_stream import RedditStream
from quant.polymarket.polymarket_stream import PolymarketStream
from quant.dashboard.live_console import LiveConsole
from quant.dashboard.json_export import JSONExporter

setup_logging("INFO")
logger = get_logger(__name__)


async def main():
    """Main async entry point for standalone CLI."""
    bus = get_event_bus()
    manager = StreamManager()
    state = QuantState()

    # Shared XAU price reference for XAUT premium calculation
    xau_price_ref = [0.0]

    async def _xau_price_tracker():
        q = await bus.subscribe(EventType.XAU_PRICE)
        while True:
            event = await q.get()
            xau_price_ref[0] = event.data.price

    logger.info("quant_engine_starting")
    print("\n" + "=" * 60)
    print("  XAU Quant Signal Engine")
    print("  Focused: XAU/USD + XAUT/USDT")
    print("  Streams: Market + News + Reddit + Polymarket")
    print("  Dashboard: Rich terminal (8 panels)")
    print("=" * 60 + "\n")

    # Instantiate all components
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
