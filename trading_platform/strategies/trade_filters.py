from __future__ import annotations

from trading_platform.strategies.signal_policy import StrategyContext, StrategyDecision
from trading_platform.utils.validation import StrategyConfig


class TradeFilterEngine:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def apply(self, decision: StrategyDecision, context: StrategyContext) -> StrategyDecision:
        if context.consecutive_losses >= self.config.max_loss_streak:
            return StrategyDecision(
                instrument=decision.instrument,
                target_fraction=0.0,
                action="hold",
                reason="loss-streak cooldown active",
                predicted_probability=decision.predicted_probability,
                expected_return=decision.expected_return,
                blocked=True,
                metadata={"cooldown_bars": self.config.cooldown_bars_after_loss_streak},
            )
        return decision
