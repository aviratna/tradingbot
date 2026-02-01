"""
Backtesting Engine for Precious Metals Trading Strategies.
Tests strategies on historical data and generates performance metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Callable
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    """Status of a backtest trade."""
    OPEN = "open"
    CLOSED_TP1 = "closed_tp1"
    CLOSED_TP2 = "closed_tp2"
    CLOSED_SL = "closed_sl"
    CLOSED_EXPIRED = "closed_expired"


@dataclass
class BacktestTrade:
    """Individual trade in backtest."""
    trade_id: int
    entry_time: datetime
    entry_price: float
    direction: str  # BUY or SELL
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    position_size: float
    status: TradeStatus = TradeStatus.OPEN
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    profit_loss: float = 0.0
    profit_pips: float = 0.0
    signal_type: str = ""
    confidence: float = 0.0


@dataclass
class BacktestMetrics:
    """Performance metrics from backtest."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    avg_trade_duration: float = 0.0  # In minutes
    best_session: str = ""
    worst_session: str = ""
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    expectancy: float = 0.0


@dataclass
class SessionPerformance:
    """Performance by trading session."""
    session: str
    trades: int
    win_rate: float
    net_profit: float
    avg_profit_per_trade: float


@dataclass
class BacktestResult:
    """Complete backtest result."""
    symbol: str
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    metrics: BacktestMetrics
    trades: list[BacktestTrade]
    equity_curve: list[float]
    session_performance: list[SessionPerformance]
    daily_returns: list[float]


class BacktestEngine:
    """
    Backtesting engine for precious metals trading strategies.

    Features:
    - Test signal generation strategies on historical data
    - Calculate comprehensive performance metrics
    - Analyze performance by trading session
    - Generate equity curves and drawdown analysis
    """

    def __init__(
        self,
        initial_balance: float = 5000,
        risk_per_trade: float = 2.0,
        commission_per_lot: float = 7.0,
        slippage_pips: float = 0.5
    ):
        """
        Initialize backtest engine.

        Args:
            initial_balance: Starting account balance
            risk_per_trade: Percentage of account to risk per trade
            commission_per_lot: Commission per lot traded
            slippage_pips: Expected slippage in pips
        """
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.commission_per_lot = commission_per_lot
        self.slippage_pips = slippage_pips

        self._trades: list[BacktestTrade] = []
        self._equity_curve: list[float] = []
        self._balance = initial_balance

    def _get_session(self, timestamp: datetime) -> str:
        """Determine trading session from timestamp."""
        hour = timestamp.hour

        if 0 <= hour < 8:
            return "ASIAN"
        elif 8 <= hour < 13:
            return "LONDON"
        elif 13 <= hour < 14:
            return "NY_OVERLAP"
        elif 14 <= hour < 20:
            return "NEW_YORK"
        else:
            return "NY_CLOSE"

    def _calculate_pip_value(self, symbol: str) -> float:
        """Calculate pip value for symbol."""
        if "XAU" in symbol:
            return 10.0  # $10 per pip for gold (0.01 lot)
        elif "XAG" in symbol:
            return 5.0  # $5 per pip for silver
        return 10.0

    def _apply_slippage(
        self,
        price: float,
        direction: str,
        is_entry: bool
    ) -> float:
        """Apply slippage to price."""
        slippage = self.slippage_pips * 0.01  # Convert pips to price

        if is_entry:
            # Entry slippage is against us
            if direction == "BUY":
                return price + slippage
            else:
                return price - slippage
        else:
            # Exit slippage is also against us
            if direction == "BUY":
                return price - slippage
            else:
                return price + slippage

    def _check_trade_exit(
        self,
        trade: BacktestTrade,
        candle: pd.Series,
        pip_value: float
    ) -> Optional[BacktestTrade]:
        """Check if trade should be closed."""
        high = candle['high']
        low = candle['low']
        timestamp = candle.name if hasattr(candle, 'name') else datetime.now(timezone.utc)

        if trade.direction == "BUY":
            # Check stop loss
            if low <= trade.stop_loss:
                exit_price = self._apply_slippage(trade.stop_loss, "BUY", False)
                pips = (exit_price - trade.entry_price) / 0.01
                profit = pips * pip_value * trade.position_size
                profit -= self.commission_per_lot * trade.position_size

                trade.status = TradeStatus.CLOSED_SL
                trade.exit_time = timestamp
                trade.exit_price = exit_price
                trade.profit_loss = profit
                trade.profit_pips = pips
                return trade

            # Check take profit 2 first (full close)
            if high >= trade.take_profit_2:
                exit_price = self._apply_slippage(trade.take_profit_2, "BUY", False)
                pips = (exit_price - trade.entry_price) / 0.01
                profit = pips * pip_value * trade.position_size
                profit -= self.commission_per_lot * trade.position_size

                trade.status = TradeStatus.CLOSED_TP2
                trade.exit_time = timestamp
                trade.exit_price = exit_price
                trade.profit_loss = profit
                trade.profit_pips = pips
                return trade

            # Check take profit 1
            if high >= trade.take_profit_1:
                exit_price = self._apply_slippage(trade.take_profit_1, "BUY", False)
                pips = (exit_price - trade.entry_price) / 0.01
                profit = pips * pip_value * trade.position_size
                profit -= self.commission_per_lot * trade.position_size

                trade.status = TradeStatus.CLOSED_TP1
                trade.exit_time = timestamp
                trade.exit_price = exit_price
                trade.profit_loss = profit
                trade.profit_pips = pips
                return trade

        else:  # SELL
            # Check stop loss
            if high >= trade.stop_loss:
                exit_price = self._apply_slippage(trade.stop_loss, "SELL", False)
                pips = (trade.entry_price - exit_price) / 0.01
                profit = pips * pip_value * trade.position_size
                profit -= self.commission_per_lot * trade.position_size

                trade.status = TradeStatus.CLOSED_SL
                trade.exit_time = timestamp
                trade.exit_price = exit_price
                trade.profit_loss = profit
                trade.profit_pips = pips
                return trade

            # Check take profit 2
            if low <= trade.take_profit_2:
                exit_price = self._apply_slippage(trade.take_profit_2, "SELL", False)
                pips = (trade.entry_price - exit_price) / 0.01
                profit = pips * pip_value * trade.position_size
                profit -= self.commission_per_lot * trade.position_size

                trade.status = TradeStatus.CLOSED_TP2
                trade.exit_time = timestamp
                trade.exit_price = exit_price
                trade.profit_loss = profit
                trade.profit_pips = pips
                return trade

            # Check take profit 1
            if low <= trade.take_profit_1:
                exit_price = self._apply_slippage(trade.take_profit_1, "SELL", False)
                pips = (trade.entry_price - exit_price) / 0.01
                profit = pips * pip_value * trade.position_size
                profit -= self.commission_per_lot * trade.position_size

                trade.status = TradeStatus.CLOSED_TP1
                trade.exit_time = timestamp
                trade.exit_price = exit_price
                trade.profit_loss = profit
                trade.profit_pips = pips
                return trade

        return None

    def _calculate_metrics(self, trades: list[BacktestTrade]) -> BacktestMetrics:
        """Calculate performance metrics from trades."""
        if not trades:
            return BacktestMetrics()

        closed_trades = [t for t in trades if t.status != TradeStatus.OPEN]

        if not closed_trades:
            return BacktestMetrics()

        winning_trades = [t for t in closed_trades if t.profit_loss > 0]
        losing_trades = [t for t in closed_trades if t.profit_loss < 0]

        total_profit = sum(t.profit_loss for t in winning_trades)
        total_loss = abs(sum(t.profit_loss for t in losing_trades))
        net_profit = total_profit - total_loss

        # Calculate metrics
        win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        avg_win = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0
        largest_win = max((t.profit_loss for t in winning_trades), default=0)
        largest_loss = min((t.profit_loss for t in losing_trades), default=0)

        # Calculate drawdown
        equity = self.initial_balance
        peak = equity
        max_dd = 0
        max_dd_percent = 0

        for trade in closed_trades:
            equity += trade.profit_loss
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_percent = (dd / peak) * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_percent = dd_percent

        # Calculate Sharpe ratio (simplified)
        returns = [t.profit_loss for t in closed_trades]
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        # Average trade duration
        durations = []
        for t in closed_trades:
            if t.entry_time and t.exit_time:
                duration = (t.exit_time - t.entry_time).total_seconds() / 60
                durations.append(duration)
        avg_duration = np.mean(durations) if durations else 0

        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0

        for t in closed_trades:
            if t.profit_loss > 0:
                current_wins += 1
                current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consec_losses = max(max_consec_losses, current_losses)

        # Expectancy
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)

        return BacktestMetrics(
            total_trades=len(closed_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=round(win_rate, 2),
            total_profit=round(total_profit, 2),
            total_loss=round(total_loss, 2),
            net_profit=round(net_profit, 2),
            profit_factor=round(profit_factor, 2),
            average_win=round(avg_win, 2),
            average_loss=round(avg_loss, 2),
            largest_win=round(largest_win, 2),
            largest_loss=round(largest_loss, 2),
            max_drawdown=round(max_dd, 2),
            max_drawdown_percent=round(max_dd_percent, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            avg_trade_duration=round(avg_duration, 2),
            consecutive_wins=max_consec_wins,
            consecutive_losses=max_consec_losses,
            expectancy=round(expectancy, 2)
        )

    def _calculate_session_performance(
        self,
        trades: list[BacktestTrade]
    ) -> list[SessionPerformance]:
        """Calculate performance by trading session."""
        sessions = {}

        for trade in trades:
            if trade.status == TradeStatus.OPEN:
                continue

            session = self._get_session(trade.entry_time) if trade.entry_time else "UNKNOWN"

            if session not in sessions:
                sessions[session] = {
                    "trades": 0,
                    "wins": 0,
                    "profit": 0
                }

            sessions[session]["trades"] += 1
            sessions[session]["profit"] += trade.profit_loss
            if trade.profit_loss > 0:
                sessions[session]["wins"] += 1

        result = []
        for session, data in sessions.items():
            win_rate = (data["wins"] / data["trades"] * 100) if data["trades"] > 0 else 0
            avg_profit = data["profit"] / data["trades"] if data["trades"] > 0 else 0

            result.append(SessionPerformance(
                session=session,
                trades=data["trades"],
                win_rate=round(win_rate, 2),
                net_profit=round(data["profit"], 2),
                avg_profit_per_trade=round(avg_profit, 2)
            ))

        return sorted(result, key=lambda x: x.net_profit, reverse=True)

    def run_backtest(
        self,
        df: pd.DataFrame,
        signal_generator: Callable,
        symbol: str = "XAU/USD",
        strategy_name: str = "Default Strategy",
        max_open_trades: int = 1
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            df: DataFrame with OHLCV data
            signal_generator: Function that takes df slice and returns signals
            symbol: Asset symbol
            strategy_name: Name of the strategy being tested
            max_open_trades: Maximum concurrent trades

        Returns:
            BacktestResult with complete analysis
        """
        if df.empty:
            return BacktestResult(
                symbol=symbol,
                strategy_name=strategy_name,
                start_date=datetime.now(timezone.utc),
                end_date=datetime.now(timezone.utc),
                initial_balance=self.initial_balance,
                final_balance=self.initial_balance,
                metrics=BacktestMetrics(),
                trades=[],
                equity_curve=[self.initial_balance],
                session_performance=[],
                daily_returns=[]
            )

        # Reset state
        self._trades = []
        self._equity_curve = [self.initial_balance]
        self._balance = self.initial_balance

        pip_value = self._calculate_pip_value(symbol)
        trade_id = 0
        open_trades: list[BacktestTrade] = []

        # Minimum lookback for signal generation
        min_lookback = 50

        for i in range(min_lookback, len(df)):
            current_candle = df.iloc[i]
            historical_data = df.iloc[:i+1]
            timestamp = current_candle.name if hasattr(current_candle, 'name') else datetime.now(timezone.utc)

            # Check existing trades for exit
            trades_to_remove = []
            for trade in open_trades:
                closed_trade = self._check_trade_exit(trade, current_candle, pip_value)
                if closed_trade:
                    self._balance += closed_trade.profit_loss
                    self._trades.append(closed_trade)
                    trades_to_remove.append(trade)

            for trade in trades_to_remove:
                open_trades.remove(trade)

            # Update equity curve
            self._equity_curve.append(self._balance)

            # Generate signals if we have room for more trades
            if len(open_trades) < max_open_trades:
                try:
                    signals = signal_generator(historical_data, symbol)

                    # Process signals
                    if signals and hasattr(signals, 'active_signals'):
                        for signal in signals.active_signals[:1]:  # Take top signal
                            if signal.confidence >= 0.5:  # Minimum confidence
                                trade_id += 1

                                entry_price = self._apply_slippage(
                                    signal.entry_price,
                                    signal.direction.value.upper(),
                                    True
                                )

                                new_trade = BacktestTrade(
                                    trade_id=trade_id,
                                    entry_time=timestamp,
                                    entry_price=entry_price,
                                    direction=signal.direction.value.upper(),
                                    stop_loss=signal.stop_loss,
                                    take_profit_1=signal.take_profit_1,
                                    take_profit_2=signal.take_profit_2,
                                    position_size=signal.position_size_lots,
                                    signal_type=signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type),
                                    confidence=signal.confidence
                                )

                                open_trades.append(new_trade)

                except Exception as e:
                    logger.warning(f"Signal generation error at index {i}: {e}")
                    continue

        # Close any remaining open trades at last price
        last_candle = df.iloc[-1]
        for trade in open_trades:
            trade.status = TradeStatus.CLOSED_EXPIRED
            trade.exit_time = last_candle.name if hasattr(last_candle, 'name') else datetime.now(timezone.utc)
            trade.exit_price = last_candle['close']

            if trade.direction == "BUY":
                pips = (trade.exit_price - trade.entry_price) / 0.01
            else:
                pips = (trade.entry_price - trade.exit_price) / 0.01

            trade.profit_pips = pips
            trade.profit_loss = pips * pip_value * trade.position_size
            trade.profit_loss -= self.commission_per_lot * trade.position_size

            self._balance += trade.profit_loss
            self._trades.append(trade)

        # Calculate metrics
        metrics = self._calculate_metrics(self._trades)
        session_performance = self._calculate_session_performance(self._trades)

        # Find best/worst sessions
        if session_performance:
            metrics.best_session = session_performance[0].session
            metrics.worst_session = session_performance[-1].session

        # Calculate daily returns
        daily_returns = []
        if len(self._equity_curve) > 1:
            prev_equity = self._equity_curve[0]
            for equity in self._equity_curve[1:]:
                daily_return = ((equity - prev_equity) / prev_equity) * 100 if prev_equity > 0 else 0
                daily_returns.append(daily_return)
                prev_equity = equity

        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy_name,
            start_date=df.index[0] if hasattr(df.index[0], 'to_pydatetime') else datetime.now(timezone.utc),
            end_date=df.index[-1] if hasattr(df.index[-1], 'to_pydatetime') else datetime.now(timezone.utc),
            initial_balance=self.initial_balance,
            final_balance=round(self._balance, 2),
            metrics=metrics,
            trades=self._trades,
            equity_curve=self._equity_curve,
            session_performance=session_performance,
            daily_returns=daily_returns
        )

    def get_backtest_summary(self, result: BacktestResult) -> dict:
        """Get a summary of backtest results for display."""
        return {
            "strategy": result.strategy_name,
            "symbol": result.symbol,
            "period": {
                "start": result.start_date.isoformat() if result.start_date else None,
                "end": result.end_date.isoformat() if result.end_date else None
            },
            "performance": {
                "initial_balance": result.initial_balance,
                "final_balance": result.final_balance,
                "net_profit": result.metrics.net_profit,
                "net_profit_percent": round(
                    (result.final_balance - result.initial_balance) / result.initial_balance * 100, 2
                ) if result.initial_balance > 0 else 0,
                "total_trades": result.metrics.total_trades,
                "win_rate": result.metrics.win_rate,
                "profit_factor": result.metrics.profit_factor,
                "max_drawdown": result.metrics.max_drawdown,
                "max_drawdown_percent": result.metrics.max_drawdown_percent,
                "sharpe_ratio": result.metrics.sharpe_ratio,
                "expectancy": result.metrics.expectancy
            },
            "trade_stats": {
                "winning_trades": result.metrics.winning_trades,
                "losing_trades": result.metrics.losing_trades,
                "average_win": result.metrics.average_win,
                "average_loss": result.metrics.average_loss,
                "largest_win": result.metrics.largest_win,
                "largest_loss": result.metrics.largest_loss,
                "consecutive_wins": result.metrics.consecutive_wins,
                "consecutive_losses": result.metrics.consecutive_losses,
                "avg_trade_duration_minutes": result.metrics.avg_trade_duration
            },
            "session_performance": [
                {
                    "session": sp.session,
                    "trades": sp.trades,
                    "win_rate": sp.win_rate,
                    "net_profit": sp.net_profit
                }
                for sp in result.session_performance
            ],
            "best_session": result.metrics.best_session,
            "worst_session": result.metrics.worst_session
        }


# Singleton instance
backtest_engine = BacktestEngine()
