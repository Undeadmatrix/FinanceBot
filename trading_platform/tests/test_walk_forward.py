from __future__ import annotations

from trading_platform.app import run_walk_forward
from trading_platform.tests.helpers import make_test_config


def test_walk_forward_generates_fold_summary(tmp_path):
    config = make_test_config(tmp_path)
    output_dir = tmp_path / "walk_forward"
    summary, _ = run_walk_forward(config, output_dir=output_dir)

    assert not summary.empty
    assert "validation_metrics.total_return" in summary.columns
    assert "test_metrics.total_return" in summary.columns
    assert (output_dir / "walk_forward_report.txt").exists()
