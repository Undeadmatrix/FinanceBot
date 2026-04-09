from __future__ import annotations

from abc import ABC, abstractmethod

from trading_platform.broker.broker_models import BrokerOrderStatus, Order


class BaseBroker(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> BrokerOrderStatus:
        raise NotImplementedError

    @abstractmethod
    def reconcile_positions(self) -> dict[str, float]:
        raise NotImplementedError
