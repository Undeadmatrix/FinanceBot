from __future__ import annotations

from trading_platform.env.tax_engine import TaxEngine
from trading_platform.tests.helpers import make_test_config


def test_tax_engine_immediate_and_accrued_modes(tmp_path):
    config = make_test_config(tmp_path)
    config.tax.mode = "accrued"
    accrued = TaxEngine(config.tax)
    event = accrued.realize(100.0, holding_days=10)
    assert event.tax_delta == 0.0
    assert event.liability_delta > 0.0

    config.tax.mode = "immediate"
    immediate = TaxEngine(config.tax)
    event_immediate = immediate.realize(100.0, holding_days=10)
    assert event_immediate.tax_delta > 0.0
    assert immediate.taxes_paid == event_immediate.tax_delta
