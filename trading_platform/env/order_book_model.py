from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OrderBookCostModel:
    spread_bps: float
    slippage_bps: float
    market_impact_bps: float

    def cost_rate(self) -> float:
        return (self.spread_bps + self.slippage_bps + self.market_impact_bps) / 10_000.0
