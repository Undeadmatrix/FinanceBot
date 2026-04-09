from __future__ import annotations

from trading_platform.broker.base_broker import BaseBroker
from trading_platform.broker.broker_models import BrokerOrderStatus, Order
from trading_platform.broker.live_guardrails import assert_paper_trading_allowed
from trading_platform.utils.validation import BrokerConfig, RiskConfig


class AlpacaAdapter(BaseBroker):
    """Paper/live adapter skeleton. Live placement is intentionally disabled by default."""

    def __init__(self, broker_config: BrokerConfig, risk_config: RiskConfig) -> None:
        assert_paper_trading_allowed(broker_config, risk_config)
        self.broker_config = broker_config

    def submit_order(self, order: Order) -> BrokerOrderStatus:
        raise NotImplementedError("Alpaca integration is a guarded skeleton in V1")

    def reconcile_positions(self) -> dict[str, float]:
        raise NotImplementedError("Alpaca position reconciliation is not implemented in V1")
