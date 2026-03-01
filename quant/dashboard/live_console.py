"""Rich terminal live dashboard with 8 panels."""

import sys
from pathlib import Path
import time
import asyncio
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant.core.logger import get_logger

logger = get_logger(__name__)

try:
    from rich.console import Console
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.columns import Columns
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logger.warning("rich_not_installed")


def _fmt_price(p: float) -> str:
    return f"${p:,.2f}"


def _fmt_pct(p: float, show_sign=True) -> str:
    sign = "+" if p >= 0 and show_sign else ""
    return f"{sign}{p:.2f}%"


def _color_pct(p: float) -> str:
    return "green" if p >= 0 else "red"


def _score_bar(score: float, width: int = 20) -> str:
    """Visual bar representing a 0-100 score."""
    filled = int(score / 100 * width)
    empty = width - filled
    color = "green" if score >= 60 else "red" if score <= 40 else "yellow"
    bar = "█" * filled + "░" * empty
    return f"[{color}]{bar}[/{color}] {score:.0f}"


def _build_header(state) -> Panel:
    """Panel 1: Header with prices and composite score."""
    xau_price = state.xau_data.price if state.xau_data else 0
    xau_chg = state.xau_data.change_pct if state.xau_data else 0
    xaut_price = state.xaut_data.price if state.xaut_data else 0
    xaut_premium = state.xaut_data.premium_discount_pct if state.xaut_data else 0

    score = state.signal_score.composite if state.signal_score else 50
    direction = state.signal_score.direction.value if state.signal_score else "NEUTRAL"
    emoji = state.signal_score.emoji if state.signal_score else "⚖️"
    regime = state.regime_snap.regime.value if state.regime_snap else "NORMAL"
    regime_color = state.regime_snap.color if state.regime_snap else "white"

    xau_color = _color_pct(xau_chg)

    t = Table.grid(expand=True, padding=(0, 2))
    t.add_column(ratio=3)
    t.add_column(ratio=3)
    t.add_column(ratio=2)
    t.add_column(ratio=2)

    t.add_row(
        Text.assemble(
            ("XAU/USD ", "bold cyan"),
            (_fmt_price(xau_price), "bold white"),
            (" ", ""),
            (_fmt_pct(xau_chg), xau_color),
        ),
        Text.assemble(
            ("XAUT/USDT ", "bold magenta"),
            (_fmt_price(xaut_price), "bold white"),
            (f" ({'+' if xaut_premium >= 0 else ''}{xaut_premium:.2f}% vs spot)", "dim"),
        ),
        Text.assemble(
            (f"{emoji} {direction}", "bold"),
            ("\n", ""),
            (f"Score: {score:.0f}/100", "dim"),
        ),
        Text.assemble(
            (f"⚡ {regime}", regime_color),
            ("\n", ""),
            (time.strftime("%H:%M:%S"), "dim"),
        ),
    )

    return Panel(t, title="[bold]🏆 XAU Quant Signal Engine[/bold]", border_style="cyan", height=5)


def _build_technicals(state) -> Panel:
    """Panel 2: Technical indicators."""
    tech = state.tech_snap

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", expand=True)
    t.add_column("Indicator", style="dim", width=16)
    t.add_column("Value", justify="right")
    t.add_column("Signal", justify="center")

    if tech:
        t.add_row("Price", _fmt_price(tech.price), "")
        t.add_row("EMA 20", _fmt_price(tech.ema20),
                  "[green]↑ above[/green]" if tech.price > tech.ema20 else "[red]↓ below[/red]")
        t.add_row("EMA 50", _fmt_price(tech.ema50),
                  "[green]↑ above[/green]" if tech.price > tech.ema50 else "[red]↓ below[/red]")
        t.add_row("RSI 14", f"{tech.rsi14:.1f}",
                  f"[{'red' if tech.rsi_signal == 'overbought' else 'green' if tech.rsi_signal == 'oversold' else 'yellow'}]{tech.rsi_signal}[/]")
        t.add_row("VWAP", _fmt_price(tech.vwap),
                  "[green]above[/green]" if tech.price > tech.vwap else "[red]below[/red]")
        t.add_row("ATR 14", _fmt_price(tech.atr14), "")
        t.add_row("MACD", f"{tech.macd:.2f}/{tech.macd_signal:.2f}",
                  f"[{'green' if tech.macd_signal_dir == 'bullish' else 'red' if tech.macd_signal_dir == 'bearish' else 'yellow'}]{tech.macd_signal_dir}[/]")
        t.add_row("Vol Delta", f"{tech.volume_delta:+.3f}",
                  "[green]buying[/green]" if tech.volume_delta > 0.1 else "[red]selling[/red]" if tech.volume_delta < -0.1 else "neutral")
        t.add_row("Trend", tech.trend,
                  f"[{'green' if tech.trend == 'uptrend' else 'red' if tech.trend == 'downtrend' else 'yellow'}]●[/]")
        t.add_row("Tech Score", _score_bar(tech.score), "")
    else:
        t.add_row("[dim]Waiting for data...[/dim]", "", "")

    return Panel(t, title="[bold]📊 Technical Analysis[/bold]", border_style="blue")


def _build_macro(state) -> Panel:
    """Panel 3: Macro environment."""
    macro = state.macro_snap

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", expand=True)
    t.add_column("Asset", style="dim", width=10)
    t.add_column("Price", justify="right")
    t.add_column("5d Chg", justify="right")
    t.add_column("Signal", justify="center")

    if macro:
        assets = [
            ("DXY", macro.dxy_price, macro.dxy_5d_change, "inv"),   # inv = inverted for gold
            ("US10Y%", macro.us10y_price, None, "yield"),
            ("VIX", macro.vix_price, None, "fear"),
            ("SPY 5d", None, macro.spy_5d_change, "risk"),
        ]
        for name, price, chg, kind in assets:
            price_str = f"{price:.2f}" if price else "N/A"
            chg_str = _fmt_pct(chg) if chg is not None else "—"
            chg_color = _color_pct(-chg if kind == "inv" else chg) if chg is not None else "white"

            if kind == "fear" and price:
                sig = "[red]extreme[/red]" if price > 35 else "[orange1]elevated[/orange1]" if price > 25 else "[green]calm[/green]"
            elif kind == "yield" and price:
                sig = "[red]hawkish[/red]" if price > 4.5 else "[yellow]neutral[/yellow]" if price > 3.5 else "[green]dovish[/green]"
            elif chg is not None:
                sig = f"[{chg_color}]{_fmt_pct(chg)}[/{chg_color}]"
            else:
                sig = "—"

            t.add_row(name, price_str, f"[{chg_color}]{chg_str}[/{chg_color}]", sig)

        t.add_row("", "", "", "")
        t.add_row("Macro Score", _score_bar(macro.score), "", "")
        if macro.key_drivers:
            t.add_row("[dim]" + macro.key_drivers[0] + "[/dim]", "", "", "")
    else:
        t.add_row("[dim]Loading macro data...[/dim]", "", "", "")

    return Panel(t, title="[bold]🌍 Macro Environment[/bold]", border_style="yellow")


def _build_sentiment(state) -> Panel:
    """Panel 4: Sentiment."""
    sent = state.sentiment_snap

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", expand=True)
    t.add_column("Source", style="dim", width=12)
    t.add_column("Score", justify="right")
    t.add_column("Items", justify="right")
    t.add_column("Signal", justify="center")

    if sent:
        def _sent_sig(s):
            if s > 0.1:
                return "[green]bullish[/green]"
            elif s < -0.1:
                return "[red]bearish[/red]"
            return "[yellow]neutral[/yellow]"

        t.add_row("Finance News", f"{sent.news_sentiment:+.3f}", str(sent.news_item_count), _sent_sig(sent.news_sentiment))
        t.add_row("Reddit", f"{sent.reddit_sentiment:+.3f}", str(sent.reddit_item_count), _sent_sig(sent.reddit_sentiment))
        t.add_row("Geopolitical", f"{sent.geo_sentiment:+.3f}", str(sent.geo_item_count), _sent_sig(sent.geo_sentiment))
        t.add_row("", "", "", "")
        t.add_row("Sent Score", _score_bar(sent.score), "", "")
        t.add_row("Overall", "", "", f"[{'green' if sent.overall_sentiment == 'bullish' else 'red' if sent.overall_sentiment == 'bearish' else 'yellow'}]{sent.overall_sentiment}[/]")
    else:
        t.add_row("[dim]Collecting sentiment...[/dim]", "", "", "")

    return Panel(t, title="[bold]💬 Sentiment & Social[/bold]", border_style="magenta")


def _build_polymarket(state) -> Panel:
    """Panel 5: Polymarket intel."""
    poly = state.poly_data

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", expand=True)
    t.add_column("Market", no_wrap=False, ratio=5)
    t.add_column("Yes%", justify="right", ratio=1)
    t.add_column("Signal", justify="center", ratio=2)

    if poly:
        for evt in (poly.metals_events + poly.geo_events)[:6]:
            title = evt.get("title", "")[:40]
            yes_prob = evt.get("yes_probability", 0.5)
            signal = evt.get("price_signal", "neutral")
            sig_color = "green" if signal == "bullish" else "red" if signal == "bearish" else "yellow"
            t.add_row(title, f"{yes_prob*100:.1f}%", f"[{sig_color}]{signal}[/{sig_color}]")

        t.add_row("", "", "")
        bias = poly.overall_bias
        bias_color = "green" if bias == "bullish" else "red" if bias == "bearish" else "yellow"
        t.add_row(
            f"Risk-Off Index: {poly.risk_off_index:.2f}",
            "",
            f"[{bias_color}]{bias.upper()}[/{bias_color}]",
        )
    else:
        t.add_row("[dim]Loading Polymarket...[/dim]", "", "")

    return Panel(t, title="[bold]🎯 Polymarket Intel[/bold]", border_style="cyan")


def _build_correlations(state) -> Panel:
    """Panel 6: Correlation matrix."""
    corr = state.corr_snap

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", expand=True)
    t.add_column("XAU vs", style="dim", width=10)
    t.add_column("Correlation", justify="right")
    t.add_column("Interpretation", justify="left")

    if corr:
        pairs = [
            ("DXY", corr.xau_dxy),
            ("SPY", corr.xau_spy),
            ("VIX", corr.xau_vix),
            ("USO", corr.xau_uso),
            ("XAUT", corr.xau_xaut),
        ]
        for name, val in pairs:
            if val is None:
                t.add_row(name, "N/A", "insufficient data")
                continue
            bar = "▓" * int(abs(val) * 10) + "░" * (10 - int(abs(val) * 10))
            color = "red" if val < -0.3 else "green" if val > 0.3 else "yellow"
            interp = (
                "inverse (expected)" if name == "DXY" and val < -0.3
                else "positive" if val > 0.3
                else "negative" if val < -0.3
                else "uncorrelated"
            )
            t.add_row(name, f"[{color}]{val:+.3f}[/{color}] {bar}", interp)

        if corr.regime_shift_detected:
            t.add_row("", "", "")
            t.add_row("⚠️ REGIME SHIFT", "", f"[bold yellow]{corr.shift_description[:30]}[/bold yellow]")
        t.add_row("[dim]", f"n={corr.data_points}", "[/dim]")
    else:
        t.add_row("[dim]Building correlation history...[/dim]", "", "")

    return Panel(t, title="[bold]🔗 Correlations (30d)[/bold]", border_style="blue")


def _build_trade(state) -> Panel:
    """Panel 7: Trade suggestion."""
    trade = state.trade_suggestion
    risk = state.risk_snap

    t = Table.grid(expand=True, padding=(0, 1))
    t.add_column(ratio=1)
    t.add_column(ratio=1)

    if trade:
        action_color = "bright_green" if trade.action == "BUY" else "bright_red" if trade.action == "SELL" else "yellow"

        t.add_row(
            Text.assemble(
                (f"● {trade.action}", f"bold {action_color}"),
                (f"  Score: {trade.signal_score:.0f}/100\n", "dim"),
                (f"Entry:  ${trade.entry_low:,.2f} – ${trade.entry_high:,.2f}\n", "white"),
                (f"Stop:   ${trade.stop_loss:,.2f}\n", "red"),
                (f"TP1:    ${trade.take_profit_1:,.2f}\n", "green"),
                (f"TP2:    ${trade.take_profit_2:,.2f}\n", "bright_green"),
                (f"R:R     {trade.r_r_ratio:.2f}:1\n", "cyan"),
            ),
            Text.assemble(
                (f"ATR:    ${trade.atr14:,.2f}\n", "dim"),
                (f"Size:   {trade.position_size_factor:.0%}\n", "dim"),
                (f"Regime: {trade.regime}\n", "yellow"),
            ),
        )
        for reason in trade.rationale[:3]:
            t.add_row(Text(f"  {reason[:60]}", style="dim"), "")
    elif risk:
        t.add_row(
            Text.assemble(
                (f"Vol: {risk.vol_regime.value}  ATR: ${risk.atr14:,.2f}\n", ""),
                (f"ATR%ile: {risk.atr_percentile:.0f}th\n", "dim"),
                (f"Size factor: {risk.position_size_factor:.0%}\n", ""),
                ("Awaiting signal threshold...", "dim"),
            ),
            "",
        )
    else:
        t.add_row(Text("[dim]Waiting for risk data...[/dim]"), "")

    return Panel(t, title="[bold]💼 Trade Suggestion[/bold]", border_style="green")


def _build_events_log(state) -> Panel:
    """Panel 8: Recent events log."""
    events = state.recent_events[-12:]  # last 12 events

    t = Table(box=box.SIMPLE, show_header=False, expand=True)
    t.add_column("Time", style="dim", width=8)
    t.add_column("Event", ratio=1)

    if events:
        for ts, msg, color in reversed(events):
            t.add_row(
                time.strftime("%H:%M:%S", time.localtime(ts)),
                Text(msg[:80], style=color or "white"),
            )
    else:
        t.add_row("", "[dim]Waiting for events...[/dim]")

    return Panel(t, title="[bold]📡 Live Events[/bold]", border_style="dim")


def _build_forecast(state) -> Panel:
    """Forecast panel (embedded in trade panel)."""
    fc = state.forecast_snap

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", expand=True)
    t.add_column("Horizon", width=8)
    t.add_column("Forecast", justify="right")
    t.add_column("Range", justify="right")

    if fc:
        for sc in fc.scenarios:
            diff_pct = (sc.base_forecast - fc.current_price) / fc.current_price * 100
            color = "green" if diff_pct > 0 else "red"
            t.add_row(
                f"{sc.horizon_minutes}min",
                f"[{color}]{_fmt_price(sc.base_forecast)}[/{color}]",
                f"{_fmt_price(sc.lower_band)}–{_fmt_price(sc.upper_band)}",
            )
        t.add_row("Method", fc.method_used, f"conf: {fc.confidence:.0%}")
    else:
        t.add_row("[dim]Building forecast...[/dim]", "", "")

    return Panel(t, title="[bold]🔮 Price Forecast[/bold]", border_style="blue")


class LiveConsole:
    """Rich Live terminal dashboard with 8 panels."""

    def __init__(self, state):
        self._state = state
        self._refresh_rate = 2  # Hz

    def _build_layout(self) -> Layout:
        """Build the full 8-panel layout."""
        state = self._state
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="body"),
            Layout(name="footer", size=14),
        )

        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="center", ratio=2),
            Layout(name="right", ratio=2),
        )

        layout["left"].split_column(
            Layout(name="technicals"),
            Layout(name="correlations"),
        )

        layout["center"].split_column(
            Layout(name="macro"),
            Layout(name="sentiment"),
        )

        layout["right"].split_column(
            Layout(name="polymarket"),
            Layout(name="forecast"),
        )

        layout["header"].update(_build_header(state))
        layout["technicals"].update(_build_technicals(state))
        layout["correlations"].update(_build_correlations(state))
        layout["macro"].update(_build_macro(state))
        layout["sentiment"].update(_build_sentiment(state))
        layout["polymarket"].update(_build_polymarket(state))
        layout["forecast"].update(_build_forecast(state))
        layout["footer"].split_row(
            Layout(_build_trade(state), ratio=2),
            Layout(_build_events_log(state), ratio=1),
        )

        return layout

    async def run(self) -> None:
        """Run the Rich Live dashboard."""
        if not RICH_AVAILABLE:
            logger.error("rich_not_available_cannot_start_dashboard")
            while True:
                await asyncio.sleep(5)
                logger.info(
                    "status_update",
                    xau=self._state.xau_data.price if self._state.xau_data else None,
                    signal=self._state.signal_score.direction.value if self._state.signal_score else None,
                )

        console = Console()
        with Live(self._build_layout(), console=console, refresh_per_second=self._refresh_rate,
                  screen=True) as live:
            logger.info("dashboard_started")
            while True:
                try:
                    live.update(self._build_layout())
                    await asyncio.sleep(1.0 / self._refresh_rate)
                except asyncio.CancelledError:
                    logger.info("dashboard_stopped")
                    break
                except Exception as e:
                    logger.error("dashboard_error", error=str(e))
                    await asyncio.sleep(1)
