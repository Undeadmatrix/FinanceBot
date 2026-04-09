from __future__ import annotations

from trading_platform.env.constraints import clip_fraction
from trading_platform.strategies.signal_policy import StrategyContext, StrategyDecision
from trading_platform.utils.validation import RiskConfig


class PositionSizer:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def size(self, decision: StrategyDecision, context: StrategyContext) -> StrategyDecision:
        if decision.blocked:
            return decision

        if self.config.sizing_mode == "volatility_adjusted":
            realized_vol = max(context.realized_volatility, 1e-4)
            scale = min(1.0, self.config.target_volatility / realized_vol)
            target_fraction = decision.target_fraction * scale
        else:
            target_fraction = decision.target_fraction * self.config.fixed_fraction

        decision.target_fraction = clip_fraction(target_fraction, self.config.max_position_fraction)
        return decision
