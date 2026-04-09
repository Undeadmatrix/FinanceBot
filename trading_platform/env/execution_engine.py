from __future__ import annotations

import logging

import pandas as pd

from trading_platform.broker.broker_models import FillEvent, Order
from trading_platform.utils.validation import ExecutionConfig


LOGGER = logging.getLogger(__name__)


class ExecutionEngine:
    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config
        self.logger = LOGGER

    def execute_market_order(self, order: Order, bar: pd.Series) -> FillEvent:
        base_price = float(bar["open"])
        spread_bps = float(bar.get("spread_bps", self.config.spread_bps))
        side_sign = 1.0 if order.side == "buy" else -1.0

        spread_component = base_price * (spread_bps / 20_000.0) * side_sign
        slippage_component = base_price * (self.config.slippage_bps / 10_000.0) * side_sign
        impact_component = base_price * (self.config.market_impact_bps / 10_000.0) * side_sign
        fill_price = base_price + spread_component + slippage_component + impact_component

        filled_quantity = order.quantity if not self.config.allow_partial_fills else order.quantity * 0.9
        gross_notional = filled_quantity * fill_price
        fees = self.config.commission_per_order + gross_notional * (self.config.fee_bps / 10_000.0)
        fill = FillEvent(
            timestamp=pd.Timestamp(order.timestamp),
            instrument=order.instrument,
            side=order.side,
            requested_quantity=order.quantity,
            filled_quantity=filled_quantity,
            fill_price=fill_price,
            gross_notional=gross_notional,
            fees=fees,
            spread_cost=abs(spread_component) * filled_quantity,
            slippage_cost=abs(slippage_component) * filled_quantity,
            market_impact_cost=abs(impact_component) * filled_quantity,
            status="partially_filled" if filled_quantity != order.quantity else "filled",
            order_type=order.order_type,
            metadata=dict(order.metadata),
        )
        self.logger.info("Executed %s %s qty=%.4f at %.4f", order.side, order.instrument, filled_quantity, fill_price)
        return fill
