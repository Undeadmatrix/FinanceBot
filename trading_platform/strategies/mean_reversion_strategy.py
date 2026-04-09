from __future__ import annotations

from trading_platform.strategies.signal_policy import StrategyDecision


class MeanReversionStrategy:
    def generate_decision(self, row: dict[str, float], instrument: str) -> StrategyDecision:
        zscore = float(row.get("zscore_20", 0.0))
        target = 1.0 if zscore < -1.0 else 0.0
        return StrategyDecision(
            instrument=instrument,
            target_fraction=target,
            action="buy" if target > 0 else "hold",
            reason="mean reversion z-score baseline",
            predicted_probability=0.5 + max(min(-zscore * 0.1, 0.49), -0.49),
            expected_return=max(0.0, -zscore * 0.002),
        )
