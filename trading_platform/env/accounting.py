from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class TradeRecord:
    timestamp: pd.Timestamp
    instrument: str
    side: str
    quantity: float
    fill_price: float
    gross_notional: float
    fees: float
    realized_pnl: float
    taxes_paid: float
    tax_liability_delta: float


@dataclass(slots=True)
class PortfolioSnapshot:
    timestamp: pd.Timestamp
    cash: float
    market_value: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float
    taxes_paid: float
    tax_liability: float
    gross_exposure: float
    turnover: float
    drawdown: float
    positions: dict[str, float] = field(default_factory=dict)
