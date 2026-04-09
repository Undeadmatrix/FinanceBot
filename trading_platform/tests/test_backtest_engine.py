from __future__ import annotations

from trading_platform.backtest.engine import BacktestEngine
from trading_platform.app import build_dataset_bundle
from trading_platform.tests.helpers import make_test_config


def test_backtest_engine_runs_sequentially_and_records_metrics(tmp_path):
    config = make_test_config(tmp_path)
    config.backtest.benchmark = []
    bundle = build_dataset_bundle(config)
    engine = BacktestEngine(
        backtest_config=config.backtest,
        execution_config=config.execution,
        tax_config=config.tax,
        risk_config=config.risk,
        broker_config=config.broker,
        strategy_config=config.strategy,
        model_config=config.model,
    )
    result = engine.run(bundle, include_benchmarks=False)

    assert not result.snapshots.empty
    assert "total_return" in result.metrics
    assert result.signals["execution_timestamp"].min() > result.signals["timestamp"].min()
    if not result.trades.empty:
        assert result.trades["timestamp"].min() >= result.signals["execution_timestamp"].min()
