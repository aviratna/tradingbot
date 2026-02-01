"""
Profit & Loss Tracker for Precious Metals Trading.
Tracks daily, weekly, and monthly P&L with performance metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, date
from enum import Enum
from typing import Optional
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TradeResult(Enum):
    """Trade result types."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


@dataclass
class Trade:
    """Individual trade record."""
    trade_id: str
    symbol: str
    direction: str  # BUY or SELL
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    lots: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    profit_loss: float = 0.0
    profit_pips: float = 0.0
    result: Optional[TradeResult] = None
    signal_type: str = ""
    notes: str = ""


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    win_rate: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0


@dataclass
class WeeklyStats:
    """Weekly trading statistics."""
    week_start: date
    week_end: date
    trades: int = 0
    wins: int = 0
    net_pnl: float = 0.0
    win_rate: float = 0.0
    daily_stats: list[DailyStats] = field(default_factory=list)


@dataclass
class MonthlyStats:
    """Monthly trading statistics."""
    month: int
    year: int
    trades: int = 0
    wins: int = 0
    net_pnl: float = 0.0
    win_rate: float = 0.0
    weekly_stats: list[WeeklyStats] = field(default_factory=list)
    best_day: Optional[DailyStats] = None
    worst_day: Optional[DailyStats] = None


@dataclass
class PerformanceMetrics:
    """Overall performance metrics."""
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_rr: float = 0.0
    expectancy: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    best_streak: int = 0
    worst_streak: int = 0
    current_streak: int = 0
    streak_type: str = ""  # WIN or LOSS


class PnLTracker:
    """
    Tracks profit and loss for precious metals trading.

    Features:
    - Real-time P&L tracking
    - Daily/weekly/monthly summaries
    - Trade journal with notes
    - Performance metrics calculation
    - Persistence to JSON file
    """

    def __init__(
        self,
        initial_balance: float = 5000,
        data_file: Optional[str] = None
    ):
        """
        Initialize P&L tracker.

        Args:
            initial_balance: Starting account balance
            data_file: Path to JSON file for persistence
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.data_file = data_file or "pnl_data.json"

        self._trades: list[Trade] = []
        self._daily_stats: dict[date, DailyStats] = {}
        self._peak_balance = initial_balance

        # Load existing data if available
        self._load_data()

    def _load_data(self) -> None:
        """Load persisted data from file."""
        try:
            path = Path(self.data_file)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.current_balance = data.get('current_balance', self.initial_balance)
                    self._peak_balance = data.get('peak_balance', self.initial_balance)

                    # Load trades
                    for trade_data in data.get('trades', []):
                        trade = Trade(
                            trade_id=trade_data['trade_id'],
                            symbol=trade_data['symbol'],
                            direction=trade_data['direction'],
                            entry_time=datetime.fromisoformat(trade_data['entry_time']),
                            entry_price=trade_data['entry_price'],
                            exit_time=datetime.fromisoformat(trade_data['exit_time']) if trade_data.get('exit_time') else None,
                            exit_price=trade_data.get('exit_price'),
                            lots=trade_data.get('lots', 0),
                            profit_loss=trade_data.get('profit_loss', 0),
                            profit_pips=trade_data.get('profit_pips', 0),
                            result=TradeResult(trade_data['result']) if trade_data.get('result') else None,
                            signal_type=trade_data.get('signal_type', ''),
                            notes=trade_data.get('notes', '')
                        )
                        self._trades.append(trade)

                logger.info(f"Loaded {len(self._trades)} trades from {self.data_file}")

        except Exception as e:
            logger.warning(f"Could not load P&L data: {e}")

    def _save_data(self) -> None:
        """Save data to file."""
        try:
            data = {
                'initial_balance': self.initial_balance,
                'current_balance': self.current_balance,
                'peak_balance': self._peak_balance,
                'trades': [
                    {
                        'trade_id': t.trade_id,
                        'symbol': t.symbol,
                        'direction': t.direction,
                        'entry_time': t.entry_time.isoformat(),
                        'entry_price': t.entry_price,
                        'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                        'exit_price': t.exit_price,
                        'lots': t.lots,
                        'profit_loss': t.profit_loss,
                        'profit_pips': t.profit_pips,
                        'result': t.result.value if t.result else None,
                        'signal_type': t.signal_type,
                        'notes': t.notes
                    }
                    for t in self._trades
                ]
            }

            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Could not save P&L data: {e}")

    def record_trade(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        lots: float,
        profit_loss: float,
        profit_pips: float,
        entry_time: Optional[datetime] = None,
        exit_time: Optional[datetime] = None,
        signal_type: str = "",
        notes: str = ""
    ) -> Trade:
        """
        Record a completed trade.

        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            direction: BUY or SELL
            entry_price: Entry price
            exit_price: Exit price
            lots: Position size in lots
            profit_loss: Profit/loss in USD
            profit_pips: Profit/loss in pips
            entry_time: Trade entry time
            exit_time: Trade exit time
            signal_type: Type of signal that triggered the trade
            notes: Additional notes

        Returns:
            Recorded Trade object
        """
        entry_time = entry_time or datetime.now(timezone.utc)
        exit_time = exit_time or datetime.now(timezone.utc)

        # Determine result
        if profit_loss > 0:
            result = TradeResult.WIN
        elif profit_loss < 0:
            result = TradeResult.LOSS
        else:
            result = TradeResult.BREAKEVEN

        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_time=entry_time,
            entry_price=entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            lots=lots,
            profit_loss=profit_loss,
            profit_pips=profit_pips,
            result=result,
            signal_type=signal_type,
            notes=notes
        )

        self._trades.append(trade)

        # Update balance
        self.current_balance += profit_loss
        if self.current_balance > self._peak_balance:
            self._peak_balance = self.current_balance

        # Update daily stats
        trade_date = exit_time.date()
        self._update_daily_stats(trade_date, trade)

        # Save data
        self._save_data()

        return trade

    def _update_daily_stats(self, trade_date: date, trade: Trade) -> None:
        """Update daily statistics with new trade."""
        if trade_date not in self._daily_stats:
            self._daily_stats[trade_date] = DailyStats(date=trade_date)

        stats = self._daily_stats[trade_date]
        stats.trades += 1

        if trade.result == TradeResult.WIN:
            stats.wins += 1
            stats.gross_profit += trade.profit_loss
            if trade.profit_loss > stats.best_trade:
                stats.best_trade = trade.profit_loss
        elif trade.result == TradeResult.LOSS:
            stats.losses += 1
            stats.gross_loss += abs(trade.profit_loss)
            if trade.profit_loss < stats.worst_trade:
                stats.worst_trade = trade.profit_loss

        stats.net_pnl = stats.gross_profit - stats.gross_loss
        stats.win_rate = (stats.wins / stats.trades * 100) if stats.trades > 0 else 0
        stats.avg_win = stats.gross_profit / stats.wins if stats.wins > 0 else 0
        stats.avg_loss = stats.gross_loss / stats.losses if stats.losses > 0 else 0

    def get_today_stats(self) -> DailyStats:
        """Get today's trading statistics."""
        today = datetime.now(timezone.utc).date()
        if today in self._daily_stats:
            return self._daily_stats[today]
        return DailyStats(date=today)

    def get_daily_stats(self, target_date: date) -> DailyStats:
        """Get statistics for a specific date."""
        if target_date in self._daily_stats:
            return self._daily_stats[target_date]
        return DailyStats(date=target_date)

    def get_weekly_stats(self, week_start: Optional[date] = None) -> WeeklyStats:
        """Get weekly trading statistics."""
        if week_start is None:
            # Get current week (Monday start)
            today = datetime.now(timezone.utc).date()
            week_start = today - timedelta(days=today.weekday())

        week_end = week_start + timedelta(days=6)

        daily_list = []
        total_trades = 0
        total_wins = 0
        total_pnl = 0.0

        for i in range(7):
            day = week_start + timedelta(days=i)
            if day in self._daily_stats:
                daily = self._daily_stats[day]
                daily_list.append(daily)
                total_trades += daily.trades
                total_wins += daily.wins
                total_pnl += daily.net_pnl

        return WeeklyStats(
            week_start=week_start,
            week_end=week_end,
            trades=total_trades,
            wins=total_wins,
            net_pnl=round(total_pnl, 2),
            win_rate=round((total_wins / total_trades * 100) if total_trades > 0 else 0, 2),
            daily_stats=daily_list
        )

    def get_monthly_stats(
        self,
        month: Optional[int] = None,
        year: Optional[int] = None
    ) -> MonthlyStats:
        """Get monthly trading statistics."""
        if month is None or year is None:
            today = datetime.now(timezone.utc).date()
            month = today.month
            year = today.year

        # Collect all days in the month
        total_trades = 0
        total_wins = 0
        total_pnl = 0.0
        best_day = None
        worst_day = None

        for day_date, daily in self._daily_stats.items():
            if day_date.month == month and day_date.year == year:
                total_trades += daily.trades
                total_wins += daily.wins
                total_pnl += daily.net_pnl

                if best_day is None or daily.net_pnl > best_day.net_pnl:
                    best_day = daily
                if worst_day is None or daily.net_pnl < worst_day.net_pnl:
                    worst_day = daily

        return MonthlyStats(
            month=month,
            year=year,
            trades=total_trades,
            wins=total_wins,
            net_pnl=round(total_pnl, 2),
            win_rate=round((total_wins / total_trades * 100) if total_trades > 0 else 0, 2),
            best_day=best_day,
            worst_day=worst_day
        )

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Calculate overall performance metrics."""
        if not self._trades:
            return PerformanceMetrics()

        completed_trades = [t for t in self._trades if t.result is not None]
        if not completed_trades:
            return PerformanceMetrics()

        wins = [t for t in completed_trades if t.result == TradeResult.WIN]
        losses = [t for t in completed_trades if t.result == TradeResult.LOSS]

        total_profit = sum(t.profit_loss for t in wins)
        total_loss = abs(sum(t.profit_loss for t in losses))

        # Win rate
        win_rate = len(wins) / len(completed_trades) * 100 if completed_trades else 0

        # Profit factor
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Average R:R (simplified)
        avg_win = total_profit / len(wins) if wins else 0
        avg_loss = total_loss / len(losses) if losses else 0
        avg_rr = avg_win / avg_loss if avg_loss > 0 else 0

        # Expectancy
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)

        # Drawdown
        max_dd = 0
        current_dd = 0
        peak = self.initial_balance
        running_balance = self.initial_balance

        for trade in completed_trades:
            running_balance += trade.profit_loss
            if running_balance > peak:
                peak = running_balance
            dd = peak - running_balance
            if dd > max_dd:
                max_dd = dd

        current_dd = self._peak_balance - self.current_balance

        # Streaks
        best_streak = 0
        worst_streak = 0
        current_streak = 0
        streak_type = ""
        temp_streak = 0
        last_result = None

        for trade in completed_trades:
            if trade.result == last_result:
                temp_streak += 1
            else:
                if last_result == TradeResult.WIN and temp_streak > best_streak:
                    best_streak = temp_streak
                elif last_result == TradeResult.LOSS and temp_streak > worst_streak:
                    worst_streak = temp_streak
                temp_streak = 1
                last_result = trade.result

        # Current streak
        if completed_trades:
            current_result = completed_trades[-1].result
            for trade in reversed(completed_trades):
                if trade.result == current_result:
                    current_streak += 1
                else:
                    break
            streak_type = "WIN" if current_result == TradeResult.WIN else "LOSS"

        return PerformanceMetrics(
            total_trades=len(completed_trades),
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            average_rr=round(avg_rr, 2),
            expectancy=round(expectancy, 2),
            max_drawdown=round(max_dd, 2),
            current_drawdown=round(current_dd, 2),
            best_streak=max(best_streak, temp_streak if last_result == TradeResult.WIN else 0),
            worst_streak=max(worst_streak, temp_streak if last_result == TradeResult.LOSS else 0),
            current_streak=current_streak,
            streak_type=streak_type
        )

    def get_recent_trades(self, count: int = 10) -> list[Trade]:
        """Get most recent trades."""
        return sorted(
            self._trades,
            key=lambda t: t.exit_time or t.entry_time,
            reverse=True
        )[:count]

    def get_pnl_summary(self) -> dict:
        """Get P&L summary for dashboard display."""
        today = self.get_today_stats()
        weekly = self.get_weekly_stats()
        monthly = self.get_monthly_stats()
        metrics = self.get_performance_metrics()

        return {
            "balance": {
                "initial": self.initial_balance,
                "current": round(self.current_balance, 2),
                "peak": round(self._peak_balance, 2),
                "total_pnl": round(self.current_balance - self.initial_balance, 2),
                "total_pnl_percent": round(
                    (self.current_balance - self.initial_balance) / self.initial_balance * 100, 2
                ) if self.initial_balance > 0 else 0
            },
            "today": {
                "trades": today.trades,
                "wins": today.wins,
                "pnl": round(today.net_pnl, 2),
                "win_rate": round(today.win_rate, 2)
            },
            "weekly": {
                "trades": weekly.trades,
                "wins": weekly.wins,
                "pnl": weekly.net_pnl,
                "win_rate": weekly.win_rate
            },
            "monthly": {
                "trades": monthly.trades,
                "wins": monthly.wins,
                "pnl": monthly.net_pnl,
                "win_rate": monthly.win_rate
            },
            "metrics": {
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "expectancy": metrics.expectancy,
                "max_drawdown": metrics.max_drawdown,
                "current_drawdown": metrics.current_drawdown,
                "current_streak": metrics.current_streak,
                "streak_type": metrics.streak_type
            },
            "recent_trades": [
                {
                    "id": t.trade_id,
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "pnl": round(t.profit_loss, 2),
                    "result": t.result.value if t.result else "open",
                    "time": t.exit_time.isoformat() if t.exit_time else t.entry_time.isoformat()
                }
                for t in self.get_recent_trades(5)
            ]
        }

    def clear_all_data(self) -> None:
        """Clear all trading data (use with caution)."""
        self._trades = []
        self._daily_stats = {}
        self.current_balance = self.initial_balance
        self._peak_balance = self.initial_balance
        self._save_data()
        logger.info("All P&L data cleared")


# Singleton instance
pnl_tracker = PnLTracker()
