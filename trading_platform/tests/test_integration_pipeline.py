from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_platform.app import build_dataset_bundle, load_platform_config, run_backtest
from trading_platform.data.market_data_loader import MarketDataLoader
from trading_platform.data.synthetic_generator import generate_synthetic_market
from trading_platform.tests.helpers import make_test_config


def test_end_to_end_synthetic_pipeline_runs(tmp_path):
    config = make_test_config(tmp_path)
    result, output_dir = run_backtest(config, output_dir=tmp_path / "backtest")

    assert (Path(output_dir) / "backtest_equity.png").exists()
    assert "buy_and_hold" in result.benchmark_results
    assert "benchmark_total_return" in result.metrics


def test_historical_csv_backtest_integration(tmp_path):
    config = make_test_config(tmp_path)
    frame = generate_synthetic_market(config.market_data.synthetic)
    csv_path = tmp_path / "historical.csv"
    frame.to_csv(csv_path, index=False)

    csv_config = load_platform_config()
    csv_config.market_data.source = "csv"
    csv_config.market_data.csv_path = str(csv_path)
    csv_config.backtest.results_dir = str(tmp_path / "csv_backtest")
    csv_config.backtest.min_train_bars = 40
    csv_config.backtest.retrain_frequency_bars = 10
    csv_config.backtest.benchmark = ["buy_and_hold"]
    csv_config.strategy.probability_buy_threshold = 0.52
    csv_config.strategy.probability_sell_threshold = 0.48
    csv_config.strategy.expected_return_threshold = 0.0
    csv_config.strategy.confidence_threshold = 0.01
    csv_config.strategy.no_trade_band = 0.0

    loaded = MarketDataLoader(timezone="UTC").load_csv(csv_path)
    bundle = build_dataset_bundle(csv_config, frame=loaded)
    assert not bundle.dataset.empty

    result, _ = run_backtest(csv_config, output_dir=tmp_path / "csv_results")
    assert not result.snapshots.empty
    assert "buy_and_hold" in result.benchmark_results
