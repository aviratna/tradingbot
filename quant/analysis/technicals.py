"""Technical analysis module: EMA, RSI, VWAP, ATR, MACD → TechnicalSnapshot."""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TechnicalSnapshot:
    # Indicators
    price: float
    ema20: float
    ema50: float
    rsi14: float
    vwap: float
    atr14: float
    macd: float
    macd_signal: float
    macd_hist: float
    volume_delta: float     # buy vol - sell vol estimate (positive = buying pressure)

    # Derived signals
    trend: str              # uptrend / downtrend / sideways
    rsi_signal: str         # overbought / oversold / neutral
    macd_signal_dir: str    # bullish / bearish / neutral

    # Normalized score 0-100
    score: float

    # Raw data for reference
    bar_count: int = 0
    timestamp: float = field(default_factory=lambda: __import__("time").time())


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return float(atr.iloc[-1]) if not atr.empty else 0.0


def _vwap(df: pd.DataFrame) -> float:
    """VWAP over available data (intraday approximation)."""
    if "volume" not in df.columns or df["volume"].sum() == 0:
        return float(df["close"].mean())
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (typical * df["volume"]).cumsum()
    vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)
    return float(vwap_series.iloc[-1]) if not vwap_series.empty else float(df["close"].iloc[-1])


def _macd(series: pd.Series) -> tuple:
    """Returns (macd_line, signal_line, histogram)."""
    ema12 = _ema(series, 12)
    ema26 = _ema(series, 26)
    macd_line = ema12 - ema26
    signal = _ema(macd_line, 9)
    hist = macd_line - signal
    return float(macd_line.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])


def compute_technicals(bars: list) -> Optional[TechnicalSnapshot]:
    """Compute all technical indicators from a list of XAUBar objects."""
    if len(bars) < 30:
        logger.warning("insufficient_bars", count=len(bars))
        return None

    try:
        df = pd.DataFrame([
            {
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ])
        df = df.sort_values("timestamp").reset_index(drop=True)
        close = df["close"]
        price = float(close.iloc[-1])

        ema20_s = _ema(close, 20)
        ema50_s = _ema(close, 50)
        ema20 = float(ema20_s.iloc[-1])
        ema50 = float(ema50_s.iloc[-1])
        rsi14 = _rsi(close, 14)
        atr14 = _atr(df, 14)
        vwap = _vwap(df.tail(100))  # Intraday VWAP on last 100 bars
        macd_val, macd_sig_val, macd_hist_val = _macd(close)

        # Volume delta: estimate buying vs selling pressure
        # Green bar (close > open) = buying, Red bar = selling
        df["is_buy"] = df["close"] > df["open"]
        vol_buy = df["volume"].where(df["is_buy"], 0).tail(20).sum()
        vol_sell = df["volume"].where(~df["is_buy"], 0).tail(20).sum()
        vol_total = vol_buy + vol_sell
        vol_delta = ((vol_buy - vol_sell) / vol_total) if vol_total > 0 else 0.0

        # Trend determination
        if price > ema20 > ema50:
            trend = "uptrend"
        elif price < ema20 < ema50:
            trend = "downtrend"
        else:
            trend = "sideways"

        # RSI signal
        if rsi14 > 70:
            rsi_signal = "overbought"
        elif rsi14 < 30:
            rsi_signal = "oversold"
        else:
            rsi_signal = "neutral"

        # MACD signal
        if macd_val > macd_sig_val and macd_hist_val > 0:
            macd_dir = "bullish"
        elif macd_val < macd_sig_val and macd_hist_val < 0:
            macd_dir = "bearish"
        else:
            macd_dir = "neutral"

        # Composite technical score (0-100)
        score = 50.0  # start neutral

        # Trend contribution (+/-20)
        if trend == "uptrend":
            score += 20
        elif trend == "downtrend":
            score -= 20

        # RSI contribution (+/-15)
        if rsi14 < 30:
            score += 15  # oversold = bullish opportunity
        elif rsi14 < 45:
            score += 8
        elif rsi14 > 70:
            score -= 15  # overbought = bearish
        elif rsi14 > 55:
            score -= 5

        # MACD contribution (+/-10)
        if macd_dir == "bullish":
            score += 10
        elif macd_dir == "bearish":
            score -= 10

        # Volume delta contribution (+/-5)
        score += vol_delta * 5

        # Price vs VWAP (+/-5)
        if vwap > 0:
            vwap_diff_pct = (price - vwap) / vwap * 100
            if vwap_diff_pct > 0.1:
                score += 5
            elif vwap_diff_pct < -0.1:
                score -= 5

        score = float(max(0.0, min(100.0, score)))

        return TechnicalSnapshot(
            price=price,
            ema20=ema20,
            ema50=ema50,
            rsi14=rsi14,
            vwap=vwap,
            atr14=atr14,
            macd=macd_val,
            macd_signal=macd_sig_val,
            macd_hist=macd_hist_val,
            volume_delta=vol_delta,
            trend=trend,
            rsi_signal=rsi_signal,
            macd_signal_dir=macd_dir,
            score=score,
            bar_count=len(bars),
        )
    except Exception as e:
        logger.error("technicals_error", error=str(e), exc_info=True)
        return None
