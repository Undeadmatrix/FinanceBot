from __future__ import annotations

import pandas as pd

from trading_platform.broker.broker_models import Order
from trading_platform.env.execution_engine import ExecutionEngine
from trading_platform.tests.helpers import make_test_config


def test_execution_engine_applies_costs_and_side_bias(tmp_path):
    config = make_test_config(tmp_path)
    engine = ExecutionEngine(config.execution)
    bar = pd.Series({"open": 100.0, "spread_bps": 10.0})

    buy_fill = engine.execute_market_order(
        Order(timestamp=pd.Timestamp("2024-01-02", tz="UTC"), instrument="TEST_A", side="buy", quantity=10.0),
        bar,
    )
    sell_fill = engine.execute_market_order(
        Order(timestamp=pd.Timestamp("2024-01-02", tz="UTC"), instrument="TEST_A", side="sell", quantity=10.0),
        bar,
    )

    assert buy_fill.fill_price > 100.0
    assert sell_fill.fill_price < 100.0
    assert buy_fill.fees > config.execution.commission_per_order
    assert sell_fill.spread_cost > 0.0
