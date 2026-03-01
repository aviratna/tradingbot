"""JSON snapshot and CSV logging every 60 seconds."""

import sys
from pathlib import Path
import asyncio
import json
import csv
import time
from dataclasses import asdict, is_dataclass
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.config import get_config
from quant.core.logger import get_logger

logger = get_logger(__name__)

# Output paths
QUANT_DATA_DIR = Path(__file__).resolve().parents[2] / "quant" / "data"
SNAPSHOTS_DIR = QUANT_DATA_DIR / "snapshots"
SIGNALS_CSV = QUANT_DATA_DIR / "signals.csv"


def _ensure_dirs():
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    QUANT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _serialize(obj: Any) -> Any:
    """Recursively serialize dataclasses and enums for JSON."""
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    elif hasattr(obj, "value"):  # Enum
        return obj.value
    elif isinstance(obj, list):
        return [_serialize(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return obj


CSV_HEADERS = [
    "timestamp", "iso_time",
    "xau_price", "xau_change_pct",
    "xaut_price", "xaut_premium_pct",
    "composite_score", "signal_direction",
    "tech_score", "macro_score", "sentiment_score", "polymarket_score",
    "regime", "vol_regime",
    "rsi14", "ema20", "ema50", "atr14",
    "dxy", "us10y", "vix",
    "news_sentiment", "reddit_sentiment",
    "poly_risk_off", "poly_bias",
    "trade_action", "entry_mid", "stop_loss", "tp1", "rr_ratio",
    "forecast_30min",
]


def _build_csv_row(state) -> dict:
    """Extract CSV row values from QuantState."""
    row = {h: "" for h in CSV_HEADERS}
    now = time.time()
    row["timestamp"] = now
    row["iso_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

    if state.xau_data:
        row["xau_price"] = state.xau_data.price
        row["xau_change_pct"] = state.xau_data.change_pct

    if state.xaut_data:
        row["xaut_price"] = state.xaut_data.price
        row["xaut_premium_pct"] = state.xaut_data.premium_discount_pct

    if state.signal_score:
        row["composite_score"] = state.signal_score.composite
        row["signal_direction"] = state.signal_score.direction.value
        row["tech_score"] = state.signal_score.tech_score
        row["macro_score"] = state.signal_score.macro_score
        row["sentiment_score"] = state.signal_score.sentiment_score
        row["polymarket_score"] = state.signal_score.polymarket_score
        row["regime"] = state.signal_score.regime_name

    if state.tech_snap:
        row["rsi14"] = round(state.tech_snap.rsi14, 2)
        row["ema20"] = round(state.tech_snap.ema20, 2)
        row["ema50"] = round(state.tech_snap.ema50, 2)
        row["atr14"] = round(state.tech_snap.atr14, 2)

    if state.macro_snap:
        row["dxy"] = state.macro_snap.dxy_price or ""
        row["us10y"] = state.macro_snap.us10y_price or ""
        row["vix"] = state.macro_snap.vix_price or ""

    if state.sentiment_snap:
        row["news_sentiment"] = round(state.sentiment_snap.news_sentiment, 4)
        row["reddit_sentiment"] = round(state.sentiment_snap.reddit_sentiment, 4)

    if state.poly_data:
        row["poly_risk_off"] = round(state.poly_data.risk_off_index, 4)
        row["poly_bias"] = state.poly_data.overall_bias

    if state.risk_snap:
        row["vol_regime"] = state.risk_snap.vol_regime.value

    if state.trade_suggestion:
        row["trade_action"] = state.trade_suggestion.action
        row["entry_mid"] = state.trade_suggestion.entry_mid
        row["stop_loss"] = state.trade_suggestion.stop_loss
        row["tp1"] = state.trade_suggestion.take_profit_1
        row["rr_ratio"] = state.trade_suggestion.r_r_ratio

    if state.forecast_snap and len(state.forecast_snap.scenarios) > 1:
        row["forecast_30min"] = state.forecast_snap.scenarios[1].base_forecast

    return row


class JSONExporter:
    """Exports state snapshots to JSON files and CSV log every 60s."""

    def __init__(self, state):
        self._state = state
        self._csv_initialized = False
        _ensure_dirs()

    def _write_json(self, state) -> None:
        """Write full JSON snapshot."""
        try:
            fname = time.strftime("%Y%m%d_%H%M%S") + ".json"
            fpath = SNAPSHOTS_DIR / fname

            snapshot = {
                "timestamp": time.time(),
                "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "xau": _serialize(state.xau_data) if state.xau_data else None,
                "xaut": _serialize(state.xaut_data) if state.xaut_data else None,
                "technicals": _serialize(state.tech_snap) if state.tech_snap else None,
                "macro": _serialize(state.macro_snap) if state.macro_snap else None,
                "sentiment": _serialize(state.sentiment_snap) if state.sentiment_snap else None,
                "correlations": _serialize(state.corr_snap) if state.corr_snap else None,
                "regime": _serialize(state.regime_snap) if state.regime_snap else None,
                "signal_score": _serialize(state.signal_score) if state.signal_score else None,
                "risk": _serialize(state.risk_snap) if state.risk_snap else None,
                "trade_suggestion": _serialize(state.trade_suggestion) if state.trade_suggestion else None,
                "forecast": _serialize(state.forecast_snap) if state.forecast_snap else None,
            }

            with open(fpath, "w") as f:
                json.dump(snapshot, f, indent=2, default=str)

            logger.debug("json_snapshot_written", path=str(fpath))
        except Exception as e:
            logger.error("json_export_error", error=str(e))

    def _write_csv_row(self, state) -> None:
        """Append a row to signals.csv."""
        try:
            row = _build_csv_row(state)
            file_exists = SIGNALS_CSV.exists()

            with open(SIGNALS_CSV, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                if not file_exists or not self._csv_initialized:
                    writer.writeheader()
                    self._csv_initialized = True
                writer.writerow(row)

            logger.debug("csv_row_written")
        except Exception as e:
            logger.error("csv_export_error", error=str(e))

    async def run(self) -> None:
        """Export loop: write JSON + CSV every export_interval seconds."""
        config = get_config()
        logger.info("json_exporter_started", interval=config.export_interval)
        while True:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._write_json, self._state
                )
                await asyncio.get_event_loop().run_in_executor(
                    None, self._write_csv_row, self._state
                )
            except asyncio.CancelledError:
                logger.info("json_exporter_stopped")
                break
            except Exception as e:
                logger.error("json_exporter_error", error=str(e))

            await asyncio.sleep(config.export_interval)
