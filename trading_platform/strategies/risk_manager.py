from __future__ import annotations

from dataclasses import dataclass, field

from trading_platform.strategies.signal_policy import StrategyDecision
from trading_platform.utils.validation import BrokerConfig, RiskConfig


@dataclass(slots=True)
class RiskState:
    gross_exposure: float
    drawdown: float
    daily_pnl_fraction: float
    trades_today: int
    human_enabled: bool
    halted: bool = False
    reasons: list[str] = field(default_factory=list)


class RiskManager:
    def __init__(self, config: RiskConfig, broker_config: BrokerConfig) -> None:
        self.config = config
        self.broker_config = broker_config

    def evaluate(self, decision: StrategyDecision, state: RiskState) -> StrategyDecision:
        reasons: list[str] = []
        if self.config.kill_switch:
            reasons.append("kill switch enabled")
        if self.broker_config.mode in {"paper", "live"} and not state.human_enabled:
            reasons.append("human-enabled flag is required outside simulation")
        if state.trades_today >= self.config.max_trades_per_day:
            reasons.append("trade count limit reached")
        if state.daily_pnl_fraction <= -self.config.max_daily_loss_fraction:
            reasons.append("daily loss stop triggered")
        if state.drawdown <= -self.config.max_drawdown_fraction:
            reasons.append("max drawdown stop triggered")
        if abs(state.gross_exposure + decision.target_fraction) > self.config.max_gross_exposure:
            reasons.append("gross exposure limit exceeded")

        if reasons:
            decision.target_fraction = 0.0
            decision.action = "hold"
            decision.reason = "; ".join(reasons)
            decision.blocked = True
        return decision
