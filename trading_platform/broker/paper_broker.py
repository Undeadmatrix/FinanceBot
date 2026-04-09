from __future__ import annotations

import uuid

import pandas as pd

from trading_platform.broker.base_broker import BaseBroker
from trading_platform.broker.broker_models import BrokerOrderStatus, Order
from trading_platform.broker.live_guardrails import assert_paper_trading_allowed
from trading_platform.utils.validation import BrokerConfig, RiskConfig


class PaperBroker(BaseBroker):
    def __init__(self, broker_config: BrokerConfig, risk_config: RiskConfig) -> None:
        assert_paper_trading_allowed(broker_config, risk_config)
        self.orders: list[BrokerOrderStatus] = []
        self.positions: dict[str, float] = {}

    def submit_order(self, order: Order) -> BrokerOrderStatus:
        status = BrokerOrderStatus(
            order_id=str(uuid.uuid4()),
            status="filled",
            submitted_at=pd.Timestamp(order.timestamp),
            filled_at=pd.Timestamp(order.timestamp),
            message="Mock paper broker immediate fill",
        )
        signed_qty = order.quantity if order.side == "buy" else -order.quantity
        self.positions[order.instrument] = self.positions.get(order.instrument, 0.0) + signed_qty
        self.orders.append(status)
        return status

    def reconcile_positions(self) -> dict[str, float]:
        return dict(self.positions)
