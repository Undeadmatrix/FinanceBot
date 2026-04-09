from __future__ import annotations

from trading_platform.strategies.position_sizer import PositionSizer
from trading_platform.strategies.risk_manager import RiskManager, RiskState
from trading_platform.strategies.signal_policy import ProbabilitySignalPolicy, StrategyContext, StrategyDecision
from trading_platform.strategies.trade_filters import TradeFilterEngine


class AlphaModelStrategy:
    def __init__(
        self,
        policy: ProbabilitySignalPolicy,
        filters: TradeFilterEngine,
        sizer: PositionSizer,
        risk_manager: RiskManager,
    ) -> None:
        self.policy = policy
        self.filters = filters
        self.sizer = sizer
        self.risk_manager = risk_manager

    def generate_decision(self, context: StrategyContext, risk_state: RiskState) -> StrategyDecision:
        decision = self.policy.decide(context)
        decision = self.filters.apply(decision, context)
        decision = self.sizer.size(decision, context)
        decision = self.risk_manager.evaluate(decision, risk_state)
        return decision
