from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading_platform.broker.broker_models import FillEvent, PositionSnapshot
from trading_platform.env.accounting import PortfolioSnapshot, TradeRecord
from trading_platform.env.tax_engine import TaxEngine


@dataclass(slots=True)
class TaxLot:
    quantity: float
    price: float
    acquired_at: pd.Timestamp


@dataclass(slots=True)
class Position:
    instrument: str
    quantity: float = 0.0
    average_cost: float = 0.0
    market_price: float = 0.0
    lots: list[TaxLot] = field(default_factory=list)

    def market_value(self) -> float:
        return self.quantity * self.market_price

    def unrealized_pnl(self) -> float:
        return (self.market_price - self.average_cost) * self.quantity


class Portfolio:
    def __init__(self, initial_cash: float, tax_engine: TaxEngine) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.tax_engine = tax_engine
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.turnover = 0.0
        self.peak_equity = initial_cash
        self.trade_history: list[TradeRecord] = []
        self.equity_history: list[PortfolioSnapshot] = []

    def process_fill(self, fill: FillEvent) -> TradeRecord:
        position = self.positions.setdefault(fill.instrument, Position(instrument=fill.instrument))
        gross_cash_flow = fill.gross_notional if fill.side == "sell" else -fill.gross_notional
        self.cash += gross_cash_flow - fill.fees
        self.fees_paid += fill.fees
        self.turnover += abs(fill.gross_notional)

        realized_pnl = 0.0
        taxes_paid = 0.0
        tax_liability_delta = 0.0

        if fill.side == "buy":
            self._apply_buy(position, fill)
        else:
            realized_pnl = self._apply_sell(position, fill)
            self.realized_pnl += realized_pnl
            tax_event = self.tax_engine.realize(realized_pnl, holding_days=int(fill.metadata.get("holding_days", 0)))
            taxes_paid = tax_event.tax_delta
            tax_liability_delta = tax_event.liability_delta
            self.cash -= taxes_paid

        record = TradeRecord(
            timestamp=fill.timestamp,
            instrument=fill.instrument,
            side=fill.side,
            quantity=fill.filled_quantity,
            fill_price=fill.fill_price,
            gross_notional=fill.gross_notional,
            fees=fill.fees,
            realized_pnl=realized_pnl,
            taxes_paid=taxes_paid,
            tax_liability_delta=tax_liability_delta,
        )
        self.trade_history.append(record)
        return record

    def _apply_buy(self, position: Position, fill: FillEvent) -> None:
        total_cost = position.average_cost * position.quantity + fill.fill_price * fill.filled_quantity
        position.quantity += fill.filled_quantity
        position.average_cost = total_cost / position.quantity if position.quantity else 0.0
        position.lots.append(TaxLot(quantity=fill.filled_quantity, price=fill.fill_price, acquired_at=fill.timestamp))
        position.market_price = fill.fill_price

    def _apply_sell(self, position: Position, fill: FillEvent) -> float:
        quantity_to_sell = fill.filled_quantity
        if quantity_to_sell > position.quantity + 1e-9:
            raise ValueError("Attempted to sell more than current position quantity")

        realized_pnl = 0.0
        remaining = quantity_to_sell
        holding_days = 0

        while remaining > 1e-9 and position.lots:
            lot = position.lots[0]
            matched = min(remaining, lot.quantity)
            realized_pnl += (fill.fill_price - lot.price) * matched
            holding_days = max(holding_days, int((fill.timestamp - lot.acquired_at).days))
            lot.quantity -= matched
            remaining -= matched
            if lot.quantity <= 1e-9:
                position.lots.pop(0)

        position.quantity -= quantity_to_sell
        position.average_cost = (
            sum(lot.quantity * lot.price for lot in position.lots) / position.quantity if position.quantity > 1e-9 else 0.0
        )
        position.market_price = fill.fill_price
        fill.metadata["holding_days"] = holding_days
        return realized_pnl

    def mark_to_market(self, timestamp: pd.Timestamp, prices: dict[str, float]) -> PortfolioSnapshot:
        market_value = 0.0
        unrealized_pnl = 0.0
        gross_exposure = 0.0
        position_quantities: dict[str, float] = {}

        for instrument, position in self.positions.items():
            if instrument in prices:
                position.market_price = float(prices[instrument])
            market_value += position.market_value()
            unrealized_pnl += position.unrealized_pnl()
            gross_exposure += abs(position.market_value())
            if abs(position.quantity) > 1e-9:
                position_quantities[instrument] = position.quantity

        equity = self.cash + market_value - self.tax_engine.tax_liability
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = equity / self.peak_equity - 1.0 if self.peak_equity else 0.0

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            market_value=market_value,
            equity=equity,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized_pnl,
            fees_paid=self.fees_paid,
            taxes_paid=self.tax_engine.taxes_paid,
            tax_liability=self.tax_engine.tax_liability,
            gross_exposure=gross_exposure / equity if equity else 0.0,
            turnover=self.turnover / self.initial_cash if self.initial_cash else 0.0,
            drawdown=drawdown,
            positions=position_quantities,
        )
        self.equity_history.append(snapshot)
        return snapshot

    def position_snapshot(self, instrument: str) -> PositionSnapshot:
        position = self.positions.get(instrument, Position(instrument=instrument))
        return PositionSnapshot(
            instrument=instrument,
            quantity=position.quantity,
            average_cost=position.average_cost,
            market_price=position.market_price,
            market_value=position.market_value(),
            unrealized_pnl=position.unrealized_pnl(),
        )
