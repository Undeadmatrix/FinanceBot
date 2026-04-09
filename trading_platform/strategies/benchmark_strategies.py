from __future__ import annotations

import numpy as np

from trading_platform.strategies.mean_reversion_strategy import MeanReversionStrategy
from trading_platform.strategies.momentum_strategy import MomentumStrategy
from trading_platform.strategies.signal_policy import StrategyDecision


class BuyAndHoldBenchmark:
    def generate_decision(self, instrument: str) -> StrategyDecision:
        return StrategyDecision(
            instrument=instrument,
            target_fraction=1.0,
            action="buy",
            reason="buy and hold benchmark",
            predicted_probability=0.5,
            expected_return=0.0,
        )


class RandomBenchmark:
    def __init__(self, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_decision(self, instrument: str) -> StrategyDecision:
        target = float(self.rng.choice([0.0, 1.0]))
        return StrategyDecision(
            instrument=instrument,
            target_fraction=target,
            action="buy" if target > 0 else "hold",
            reason="random benchmark",
            predicted_probability=float(self.rng.uniform(0.45, 0.55)),
            expected_return=0.0,
        )


class MovingAverageCrossoverBenchmark:
    def generate_decision(self, row: dict[str, float], instrument: str) -> StrategyDecision:
        short_term = float(row.get("ma_ratio_5", 0.0))
        long_term = float(row.get("ma_ratio_20", 0.0))
        target = 1.0 if short_term > long_term else 0.0
        return StrategyDecision(
            instrument=instrument,
            target_fraction=target,
            action="buy" if target > 0 else "hold",
            reason="moving-average crossover benchmark",
            predicted_probability=0.5 + max(min((short_term - long_term) * 5.0, 0.49), -0.49),
            expected_return=max(short_term - long_term, 0.0),
        )


def build_benchmark_strategy(name: str, seed: int = 7):
    if name == "buy_and_hold":
        return BuyAndHoldBenchmark()
    if name == "random":
        return RandomBenchmark(seed=seed)
    if name == "moving_average_crossover":
        return MovingAverageCrossoverBenchmark()
    if name == "momentum":
        return MomentumStrategy()
    if name == "mean_reversion":
        return MeanReversionStrategy()
    raise ValueError(f"Unsupported benchmark strategy: {name}")
