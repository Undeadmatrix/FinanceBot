from __future__ import annotations

from pathlib import Path

from trading_platform.utils.validation import PlatformConfig


def make_test_config(tmp_path: Path | None = None) -> PlatformConfig:
    config = PlatformConfig()
    config.market_data.synthetic.instruments = ["TEST_A", "TEST_B"]
    config.market_data.synthetic.periods = 320
    config.market_data.synthetic.seed = 13
    config.features.min_history = 20
    config.strategy.probability_buy_threshold = 0.52
    config.strategy.probability_sell_threshold = 0.48
    config.strategy.expected_return_threshold = 0.0
    config.strategy.confidence_threshold = 0.01
    config.strategy.no_trade_band = 0.0
    config.risk.fixed_fraction = 0.25
    config.risk.max_position_fraction = 0.5
    config.backtest.min_train_bars = 40
    config.backtest.retrain_frequency_bars = 10
    config.walk_forward.train_bars = 80
    config.walk_forward.validation_bars = 20
    config.walk_forward.test_bars = 20
    config.walk_forward.step_bars = 20
    if tmp_path is not None:
        config.backtest.results_dir = str(tmp_path / "runs")
        config.model.model_dir = str(tmp_path / "models")
    return config
