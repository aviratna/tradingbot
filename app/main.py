"""Main FastAPI application for the trading dashboard."""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
import time
from contextlib import asynccontextmanager

from .api.routes import router as api_router
from .config import settings

# ── Quant Engine Integration ──────────────────────────────────────────────────
import sys as _sys
import logging as _logging
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

_quant_state = None
_quant_task = None
_quant_log = _logging.getLogger("app.quant_launcher")


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Startup / shutdown lifecycle — starts quant engine as background task."""
    global _quant_state, _quant_task
    print("QUANT: lifespan startup beginning...", flush=True)
    try:
        print("QUANT: importing core event_bus...", flush=True)
        from quant.core.event_bus import get_event_bus, EventType
        print("QUANT: importing core logger...", flush=True)
        from quant.core.logger import setup_logging, get_logger
        print("QUANT: importing market streams...", flush=True)
        from quant.market.xau_stream import XAUStream
        from quant.market.xaut_stream import XAUTStream
        from quant.market.macro_stream import MacroStream
        print("QUANT: importing news streams...", flush=True)
        from quant.news.finance_stream import FinanceNewsStream
        from quant.news.geopolitics_stream import GeopoliticsStream
        from quant.news.reddit_stream import RedditStream
        print("QUANT: importing polymarket stream...", flush=True)
        from quant.polymarket.polymarket_stream import PolymarketStream
        print("QUANT: importing analysis orchestrator...", flush=True)
        from quant.analysis.orchestrator import AnalysisOrchestrator
        print("QUANT: importing json exporter...", flush=True)
        from quant.dashboard.json_export import JSONExporter
        print("QUANT: importing state...", flush=True)
        from quant.state import QuantState
        print("QUANT: all imports successful!", flush=True)

        setup_logging("INFO")
        log = get_logger("app.quant_launcher")
        bus = get_event_bus()

        state = QuantState()
        _quant_state = state
        app_instance.state.quant = state  # expose on FastAPI app.state
        print("QUANT: QuantState created, setting up tasks...", flush=True)

        xau_price_ref = [0.0]

        async def _xau_tracker():
            q = await bus.subscribe(EventType.XAU_PRICE)
            while True:
                event = await q.get()
                xau_price_ref[0] = event.data.price

        async def _run_all():
            try:
                print("QUANT: _run_all started, creating stream instances...", flush=True)
                xau = XAUStream()
                xaut = XAUTStream(xau_price_ref)
                macro = MacroStream()
                finance = FinanceNewsStream()
                geo = GeopoliticsStream()
                reddit = RedditStream()
                poly = PolymarketStream()
                orch = AnalysisOrchestrator(state, bus)
                exporter = JSONExporter(state)
                print("QUANT: all stream instances created, launching gather...", flush=True)
                log.info("quant_engine_all_streams_started")
                results = await asyncio.gather(
                    xau.run(),
                    xaut.run(),
                    macro.run(),
                    finance.run(),
                    geo.run(),
                    reddit.run(),
                    poly.run(),
                    _xau_tracker(),
                    orch.run_event_consumer(),
                    exporter.run(),
                    return_exceptions=True,
                )
                # Log any exceptions returned by gather
                for i, r in enumerate(results):
                    if isinstance(r, Exception):
                        print(f"QUANT: stream[{i}] failed: {type(r).__name__}: {r}", flush=True)
            except Exception as inner_e:
                import traceback as _tb
                print(f"QUANT: _run_all exception: {type(inner_e).__name__}: {inner_e}", flush=True)
                _tb.print_exc()
                log.error("quant_gather_failed", error=str(inner_e), exc_info=True)

        _quant_task = asyncio.create_task(_run_all())
        log.info("quant_engine_task_created")
        print("QUANT: background task created successfully!", flush=True)

    except Exception as e:
        import traceback as _tb
        print(f"QUANT STARTUP FAILED: {type(e).__name__}: {e}", flush=True)
        _tb.print_exc()
        _quant_log.warning(f"Quant engine could not start (non-fatal): {e}", exc_info=True)

    yield  # FastAPI serves requests here

    # Shutdown: cancel quant task
    if _quant_task and not _quant_task.done():
        _quant_task.cancel()
        try:
            await _quant_task
        except (asyncio.CancelledError, Exception):
            pass
# ─────────────────────────────────────────────────────────────────────────────


# Create FastAPI app
app = FastAPI(
    title="Trading Dashboard",
    description="Real-time trading dashboard with market data, sentiment analysis, and Fibonacci patterns",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Include API routes
app.include_router(api_router)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the main dashboard page."""
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "title": "Trading Dashboard",
            "default_stocks": settings.DEFAULT_STOCKS,
            "default_crypto": settings.DEFAULT_CRYPTO,
            "default_forex": settings.DEFAULT_FOREX
        }
    )


@app.get("/metals", response_class=HTMLResponse)
async def metals_dashboard(request: Request):
    """Render the precious metals trading dashboard."""
    return templates.TemplateResponse(
        "metals_dashboard.html",
        {
            "request": request,
            "title": "Precious Metals Trading - XAU/USD & XAG/USD"
        }
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/quant/status")
async def quant_status(request: Request):
    """Return current quant engine state as JSON."""
    # Try app.state first, then module-level fallback
    state = getattr(request.app.state, "quant", None) or _quant_state
    if state is None:
        return JSONResponse({"status": "not_running", "error": "Quant engine not initialized"}, status_code=503)

    def _safe(obj):
        """Safely extract primitive fields from dataclass."""
        if obj is None:
            return None
        try:
            from dataclasses import fields
            result = {}
            for f in fields(obj):
                val = getattr(obj, f.name, None)
                if hasattr(val, "value"):       # Enum
                    result[f.name] = val.value
                elif isinstance(val, (int, float, str, bool, type(None))):
                    result[f.name] = val
                elif isinstance(val, list):
                    result[f.name] = [str(v)[:120] for v in val[:5]]
                else:
                    result[f.name] = str(val)[:120]
            return result
        except Exception:
            return str(obj)[:200]

    return {
        "status": "running",
        "timestamp": time.time(),
        "xau": {
            "price": state.xau_data.price if state.xau_data else None,
            "change_pct": state.xau_data.change_pct if state.xau_data else None,
        },
        "xaut": {
            "price": state.xaut_data.price if state.xaut_data else None,
            "premium_pct": state.xaut_data.premium_discount_pct if state.xaut_data else None,
        },
        "signal": {
            "composite": state.signal_score.composite if state.signal_score else None,
            "direction": state.signal_score.direction.value if state.signal_score else None,
            "tech_score": state.signal_score.tech_score if state.signal_score else None,
            "macro_score": state.signal_score.macro_score if state.signal_score else None,
            "sentiment_score": state.signal_score.sentiment_score if state.signal_score else None,
            "polymarket_score": state.signal_score.polymarket_score if state.signal_score else None,
        },
        "regime": {
            "name": state.regime_snap.regime.value if state.regime_snap else None,
            "gold_bias": state.regime_snap.gold_bias if state.regime_snap else None,
            "description": state.regime_snap.description if state.regime_snap else None,
            "triggers": state.regime_snap.triggers if state.regime_snap else [],
        },
        "technicals": _safe(state.tech_snap),
        "macro": _safe(state.macro_snap),
        "sentiment": _safe(state.sentiment_snap),
        "risk": _safe(state.risk_snap),
        "trade": {
            "action": state.trade_suggestion.action if state.trade_suggestion else None,
            "entry_low": state.trade_suggestion.entry_low if state.trade_suggestion else None,
            "entry_high": state.trade_suggestion.entry_high if state.trade_suggestion else None,
            "stop_loss": state.trade_suggestion.stop_loss if state.trade_suggestion else None,
            "take_profit_1": state.trade_suggestion.take_profit_1 if state.trade_suggestion else None,
            "take_profit_2": state.trade_suggestion.take_profit_2 if state.trade_suggestion else None,
            "r_r_ratio": state.trade_suggestion.r_r_ratio if state.trade_suggestion else None,
            "rationale": state.trade_suggestion.rationale if state.trade_suggestion else [],
        },
        "forecast": {
            "direction": state.forecast_snap.forecast_direction if state.forecast_snap else None,
            "method": state.forecast_snap.method_used if state.forecast_snap else None,
            "scenarios": [
                {
                    "horizon_min": s.horizon_minutes,
                    "base": s.base_forecast,
                    "low": s.lower_band,
                    "high": s.upper_band,
                }
                for s in state.forecast_snap.scenarios
            ] if state.forecast_snap else [],
        },
        "recent_events": [
            {"time": e[0], "msg": e[1], "color": e[2]}
            for e in (state.recent_events[-10:] if state.recent_events else [])
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
