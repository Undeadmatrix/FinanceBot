from __future__ import annotations

import pandas as pd

from trading_platform.broker.broker_models import FillEvent
from trading_platform.env.portfolio import Portfolio
from trading_platform.env.tax_engine import TaxEngine
from trading_platform.tests.helpers import make_test_config


def test_portfolio_tracks_cash_pnl_and_positions(tmp_path):
    config = make_test_config(tmp_path)
    portfolio = Portfolio(initial_cash=10_000.0, tax_engine=TaxEngine(config.tax))

    buy_fill = FillEvent(
        timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
        instrument="TEST_A",
        side="buy",
        requested_quantity=10.0,
        filled_quantity=10.0,
        fill_price=100.0,
        gross_notional=1_000.0,
        fees=5.0,
        spread_cost=1.0,
        slippage_cost=1.0,
        market_impact_cost=1.0,
    )
    portfolio.process_fill(buy_fill)
    snapshot = portfolio.mark_to_market(pd.Timestamp("2024-01-02", tz="UTC"), {"TEST_A": 102.0})
    assert portfolio.cash == 8_995.0
    assert snapshot.unrealized_pnl == 20.0

    sell_fill = FillEvent(
        timestamp=pd.Timestamp("2024-01-12", tz="UTC"),
        instrument="TEST_A",
        side="sell",
        requested_quantity=10.0,
        filled_quantity=10.0,
        fill_price=105.0,
        gross_notional=1_050.0,
        fees=5.0,
        spread_cost=1.0,
        slippage_cost=1.0,
        market_impact_cost=1.0,
    )
    trade = portfolio.process_fill(sell_fill)
    final_snapshot = portfolio.mark_to_market(pd.Timestamp("2024-01-12", tz="UTC"), {"TEST_A": 105.0})
    assert trade.realized_pnl == 50.0
    assert portfolio.positions["TEST_A"].quantity == 0.0
    assert final_snapshot.cash > 10_000.0
