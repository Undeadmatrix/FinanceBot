from __future__ import annotations

from trading_platform.app import build_dataset_bundle
from trading_platform.backtest.walk_forward import WalkForwardValidator
from trading_platform.tests.helpers import make_test_config


def test_walk_forward_generates_fold_summary(tmp_path):
    config = make_test_config(tmp_path)
    bundle = build_dataset_bundle(config)
    summary = WalkForwardValidator(config).run(bundle, output_dir=tmp_path / "walk_forward")

    assert not summary.empty
    assert "validation_metrics.total_return" in summary.columns
    assert "test_metrics.total_return" in summary.columns
