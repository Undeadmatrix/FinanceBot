from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading_platform.utils.validation import StrategyConfig


@dataclass(slots=True)
class StrategyContext:
    timestamp: pd.Timestamp
    instrument: str
    predicted_probability: float
    expected_return: float
    estimated_cost_rate: float
    realized_volatility: float
    current_fraction: float
    current_drawdown: float
    consecutive_losses: int
    trades_today: int
    human_enabled: bool


@dataclass(slots=True)
class StrategyDecision:
    instrument: str
    target_fraction: float
    action: str
    reason: str
    predicted_probability: float
    expected_return: float
    blocked: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


class ProbabilitySignalPolicy:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def decide(self, context: StrategyContext) -> StrategyDecision:
        centered_probability = context.predicted_probability - 0.5
        confidence = abs(centered_probability)
        edge_after_cost = context.expected_return - context.estimated_cost_rate - self.config.cost_buffer_bps / 10_000.0

        if confidence < self.config.confidence_threshold or abs(centered_probability) < self.config.no_trade_band:
            return StrategyDecision(
                instrument=context.instrument,
                target_fraction=context.current_fraction,
                action="hold",
                reason="probability inside no-trade band",
                predicted_probability=context.predicted_probability,
                expected_return=context.expected_return,
                blocked=True,
            )

        if edge_after_cost < self.config.expected_return_threshold:
            return StrategyDecision(
                instrument=context.instrument,
                target_fraction=context.current_fraction,
                action="hold",
                reason="edge below cost-aware threshold",
                predicted_probability=context.predicted_probability,
                expected_return=context.expected_return,
                blocked=True,
            )

        if context.predicted_probability >= self.config.probability_buy_threshold:
            signal_strength = min(1.0, confidence * 2.0)
            return StrategyDecision(
                instrument=context.instrument,
                target_fraction=signal_strength,
                action="buy",
                reason="probability exceeds buy threshold",
                predicted_probability=context.predicted_probability,
                expected_return=context.expected_return,
            )

        if not self.config.long_only and context.predicted_probability <= self.config.probability_sell_threshold:
            signal_strength = -min(1.0, confidence * 2.0)
            return StrategyDecision(
                instrument=context.instrument,
                target_fraction=signal_strength,
                action="sell",
                reason="probability exceeds sell threshold",
                predicted_probability=context.predicted_probability,
                expected_return=context.expected_return,
            )

        if self.config.long_only and context.predicted_probability <= self.config.probability_sell_threshold:
            return StrategyDecision(
                instrument=context.instrument,
                target_fraction=0.0,
                action="flatten",
                reason="weak or negative signal in long-only mode",
                predicted_probability=context.predicted_probability,
                expected_return=context.expected_return,
            )

        return StrategyDecision(
            instrument=context.instrument,
            target_fraction=context.current_fraction,
            action="hold",
            reason="signal unresolved",
            predicted_probability=context.predicted_probability,
            expected_return=context.expected_return,
            blocked=True,
        )
