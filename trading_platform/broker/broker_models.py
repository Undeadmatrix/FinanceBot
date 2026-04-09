from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


OrderSide = Literal["buy", "sell"]
OrderStatus = Literal["new", "filled", "partially_filled", "cancelled", "rejected"]


@dataclass(slots=True)
class Order:
    timestamp: pd.Timestamp
    instrument: str
    side: OrderSide
    quantity: float
    order_type: str = "market"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class FillEvent:
    timestamp: pd.Timestamp
    instrument: str
    side: OrderSide
    requested_quantity: float
    filled_quantity: float
    fill_price: float
    gross_notional: float
    fees: float
    spread_cost: float
    slippage_cost: float
    market_impact_cost: float
    status: OrderStatus = "filled"
    order_type: str = "market"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PositionSnapshot:
    instrument: str
    quantity: float
    average_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float


@dataclass(slots=True)
class BrokerOrderStatus:
    order_id: str
    status: OrderStatus
    submitted_at: pd.Timestamp
    filled_at: pd.Timestamp | None = None
    message: str | None = None
