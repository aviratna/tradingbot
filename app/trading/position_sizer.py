"""
Position Sizing and Risk Management for Precious Metals Trading.
Calculates optimal position sizes based on account risk parameters.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class PositionSize:
    """Calculated position size with risk details."""
    lots: float
    units: int
    risk_amount: float
    risk_percent: float
    stop_loss_pips: float
    potential_profit_tp1: float
    potential_profit_tp2: float
    risk_reward_tp1: float
    risk_reward_tp2: float
    margin_required: float
    leverage_used: float


@dataclass
class RiskAssessment:
    """Overall risk assessment for a trade."""
    is_acceptable: bool
    risk_level: RiskLevel
    max_position_size: float
    recommended_position_size: float
    warnings: list[str]
    score: float  # 0-100 risk score


class PositionSizer:
    """
    Calculates optimal position sizes for precious metals trading.

    Supports:
    - Fixed percentage risk per trade
    - ATR-based stop loss calculation
    - Multiple risk profiles (conservative, moderate, aggressive)
    - Margin and leverage calculations
    """

    # Pip values per standard lot (100 oz gold, 5000 oz silver)
    PIP_VALUES = {
        "XAU/USD": 10.0,   # $10 per pip per lot
        "XAG/USD": 50.0,   # $50 per pip per lot (0.01 move = 1 pip)
    }

    # Margin requirements (approximate, varies by broker)
    MARGIN_REQUIREMENTS = {
        "XAU/USD": 1000,   # $1000 per lot
        "XAG/USD": 500,    # $500 per lot
    }

    # Risk profiles
    RISK_PROFILES = {
        RiskLevel.CONSERVATIVE: {
            "risk_percent": 1.0,
            "max_daily_risk": 3.0,
            "max_position_lots": 0.5,
            "min_rr_ratio": 2.0,
        },
        RiskLevel.MODERATE: {
            "risk_percent": 2.0,
            "max_daily_risk": 6.0,
            "max_position_lots": 1.0,
            "min_rr_ratio": 1.5,
        },
        RiskLevel.AGGRESSIVE: {
            "risk_percent": 3.0,
            "max_daily_risk": 10.0,
            "max_position_lots": 2.0,
            "min_rr_ratio": 1.0,
        },
    }

    def __init__(
        self,
        account_balance: float = 5000,
        risk_level: RiskLevel = RiskLevel.MODERATE,
        leverage: float = 100
    ):
        """
        Initialize position sizer.

        Args:
            account_balance: Current account balance in USD
            risk_level: Risk profile to use
            leverage: Broker leverage (e.g., 100 = 1:100)
        """
        self.account_balance = account_balance
        self.risk_level = risk_level
        self.leverage = leverage
        self._daily_risk_used = 0.0

    @property
    def risk_profile(self) -> dict:
        """Get current risk profile settings."""
        return self.RISK_PROFILES[self.risk_level]

    def update_balance(self, new_balance: float) -> None:
        """Update account balance."""
        self.account_balance = new_balance

    def reset_daily_risk(self) -> None:
        """Reset daily risk counter (call at start of each day)."""
        self._daily_risk_used = 0.0

    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol."""
        return self.PIP_VALUES.get(symbol, 10.0)

    def _get_margin_requirement(self, symbol: str) -> float:
        """Get margin requirement per lot."""
        return self.MARGIN_REQUIREMENTS.get(symbol, 1000)

    def calculate_stop_loss_pips(
        self,
        entry_price: float,
        stop_loss: float,
        symbol: str = "XAU/USD"
    ) -> float:
        """
        Calculate stop loss distance in pips.

        For gold: 1 pip = $0.01
        For silver: 1 pip = $0.01
        """
        pip_size = 0.01  # Both gold and silver use 0.01 as pip
        return abs(entry_price - stop_loss) / pip_size

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
        symbol: str = "XAU/USD",
        custom_risk_percent: Optional[float] = None
    ) -> PositionSize:
        """
        Calculate optimal position size based on risk parameters.

        Args:
            entry_price: Trade entry price
            stop_loss: Stop loss price
            take_profit_1: First take profit target
            take_profit_2: Second take profit target
            symbol: Trading symbol
            custom_risk_percent: Override default risk percentage

        Returns:
            PositionSize with all calculations
        """
        # Get pip value and risk settings
        pip_value = self._get_pip_value(symbol)
        margin_per_lot = self._get_margin_requirement(symbol)
        risk_percent = custom_risk_percent or self.risk_profile["risk_percent"]

        # Calculate risk amount
        risk_amount = self.account_balance * (risk_percent / 100)

        # Calculate stop loss in pips
        stop_loss_pips = self.calculate_stop_loss_pips(entry_price, stop_loss, symbol)

        if stop_loss_pips <= 0:
            logger.warning("Invalid stop loss distance")
            return PositionSize(
                lots=0.01,
                units=100,
                risk_amount=0,
                risk_percent=0,
                stop_loss_pips=0,
                potential_profit_tp1=0,
                potential_profit_tp2=0,
                risk_reward_tp1=0,
                risk_reward_tp2=0,
                margin_required=0,
                leverage_used=0
            )

        # Calculate lot size
        # Lot Size = Risk Amount / (Stop Loss Pips × Pip Value)
        lots = risk_amount / (stop_loss_pips * pip_value)

        # Apply limits
        max_lots = self.risk_profile["max_position_lots"]
        lots = min(lots, max_lots)
        lots = max(0.01, round(lots, 2))  # Minimum 0.01 lot

        # Calculate units (1 lot gold = 100 oz, 1 lot silver = 5000 oz)
        units_per_lot = 100 if "XAU" in symbol else 5000
        units = int(lots * units_per_lot)

        # Calculate actual risk with rounded lot size
        actual_risk = lots * stop_loss_pips * pip_value
        actual_risk_percent = (actual_risk / self.account_balance) * 100

        # Calculate potential profits
        tp1_pips = self.calculate_stop_loss_pips(entry_price, take_profit_1, symbol)
        tp2_pips = self.calculate_stop_loss_pips(entry_price, take_profit_2, symbol)

        potential_profit_tp1 = lots * tp1_pips * pip_value
        potential_profit_tp2 = lots * tp2_pips * pip_value

        # Calculate risk:reward ratios
        rr_tp1 = tp1_pips / stop_loss_pips if stop_loss_pips > 0 else 0
        rr_tp2 = tp2_pips / stop_loss_pips if stop_loss_pips > 0 else 0

        # Calculate margin and leverage
        margin_required = lots * margin_per_lot
        notional_value = lots * entry_price * units_per_lot
        leverage_used = notional_value / margin_required if margin_required > 0 else 0

        return PositionSize(
            lots=lots,
            units=units,
            risk_amount=round(actual_risk, 2),
            risk_percent=round(actual_risk_percent, 2),
            stop_loss_pips=round(stop_loss_pips, 1),
            potential_profit_tp1=round(potential_profit_tp1, 2),
            potential_profit_tp2=round(potential_profit_tp2, 2),
            risk_reward_tp1=round(rr_tp1, 2),
            risk_reward_tp2=round(rr_tp2, 2),
            margin_required=round(margin_required, 2),
            leverage_used=round(leverage_used, 1)
        )

    def assess_risk(
        self,
        position_size: PositionSize,
        current_drawdown: float = 0
    ) -> RiskAssessment:
        """
        Assess overall risk for a proposed trade.

        Args:
            position_size: Calculated position size
            current_drawdown: Current account drawdown percentage

        Returns:
            RiskAssessment with recommendations
        """
        warnings = []
        is_acceptable = True
        score = 100

        # Check daily risk limit
        potential_daily_risk = self._daily_risk_used + position_size.risk_percent
        max_daily_risk = self.risk_profile["max_daily_risk"]

        if potential_daily_risk > max_daily_risk:
            warnings.append(f"Would exceed daily risk limit ({potential_daily_risk:.1f}% > {max_daily_risk}%)")
            is_acceptable = False
            score -= 30

        # Check risk:reward ratio
        min_rr = self.risk_profile["min_rr_ratio"]
        if position_size.risk_reward_tp1 < min_rr:
            warnings.append(f"R:R ratio too low ({position_size.risk_reward_tp1:.1f} < {min_rr})")
            score -= 20

        # Check drawdown
        if current_drawdown > 10:
            warnings.append(f"High drawdown ({current_drawdown:.1f}%) - reduce position size")
            score -= 15
        if current_drawdown > 20:
            is_acceptable = False
            score -= 20

        # Check margin usage
        margin_percent = (position_size.margin_required / self.account_balance) * 100
        if margin_percent > 50:
            warnings.append(f"High margin usage ({margin_percent:.1f}%)")
            score -= 10

        # Determine risk level
        if score >= 80:
            risk_level = RiskLevel.CONSERVATIVE
        elif score >= 60:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.AGGRESSIVE

        # Calculate recommended position size
        recommended_lots = position_size.lots
        if current_drawdown > 10:
            recommended_lots *= 0.75
        if current_drawdown > 20:
            recommended_lots *= 0.5

        return RiskAssessment(
            is_acceptable=is_acceptable,
            risk_level=risk_level,
            max_position_size=self.risk_profile["max_position_lots"],
            recommended_position_size=round(recommended_lots, 2),
            warnings=warnings,
            score=max(0, score)
        )

    def calculate_for_target_profit(
        self,
        target_profit: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        symbol: str = "XAU/USD"
    ) -> PositionSize:
        """
        Calculate position size needed for target profit.

        Args:
            target_profit: Desired profit in USD
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            symbol: Trading symbol

        Returns:
            PositionSize that would achieve target profit
        """
        pip_value = self._get_pip_value(symbol)
        tp_pips = self.calculate_stop_loss_pips(entry_price, take_profit, symbol)

        if tp_pips <= 0:
            return self.calculate_position_size(
                entry_price, stop_loss, take_profit, take_profit, symbol
            )

        # Required lots = Target Profit / (TP Pips × Pip Value)
        required_lots = target_profit / (tp_pips * pip_value)
        required_lots = min(required_lots, self.risk_profile["max_position_lots"])
        required_lots = max(0.01, round(required_lots, 2))

        # Calculate with required lots
        sl_pips = self.calculate_stop_loss_pips(entry_price, stop_loss, symbol)
        actual_risk = required_lots * sl_pips * pip_value
        risk_percent = (actual_risk / self.account_balance) * 100

        return PositionSize(
            lots=required_lots,
            units=int(required_lots * (100 if "XAU" in symbol else 5000)),
            risk_amount=round(actual_risk, 2),
            risk_percent=round(risk_percent, 2),
            stop_loss_pips=round(sl_pips, 1),
            potential_profit_tp1=round(target_profit, 2),
            potential_profit_tp2=round(target_profit * 1.5, 2),
            risk_reward_tp1=round(tp_pips / sl_pips if sl_pips > 0 else 0, 2),
            risk_reward_tp2=round((tp_pips * 1.5) / sl_pips if sl_pips > 0 else 0, 2),
            margin_required=round(required_lots * self._get_margin_requirement(symbol), 2),
            leverage_used=0
        )

    def get_position_summary(self, position: PositionSize) -> dict:
        """Get position size summary for display."""
        return {
            "lots": position.lots,
            "units": position.units,
            "risk": {
                "amount": position.risk_amount,
                "percent": position.risk_percent,
                "stop_loss_pips": position.stop_loss_pips
            },
            "potential_profit": {
                "tp1": position.potential_profit_tp1,
                "tp2": position.potential_profit_tp2
            },
            "risk_reward": {
                "tp1": position.risk_reward_tp1,
                "tp2": position.risk_reward_tp2
            },
            "margin": {
                "required": position.margin_required,
                "leverage": position.leverage_used
            }
        }


# Singleton instance with default settings
position_sizer = PositionSizer()
