from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:d}"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _format_percentage(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return str(value)


def _metric_line(name: str, value: Any) -> str:
    percent_like = {"total_return", "cagr", "volatility", "max_drawdown", "fee_drag", "tax_drag", "win_rate", "exposure"}
    rendered = _format_percentage(value) if name in percent_like else _format_value(value)
    return f"{name}: {rendered}"


def write_backtest_report(
    output_dir: str | Path,
    metrics: dict[str, Any],
    trades: pd.DataFrame,
    snapshots: pd.DataFrame,
    benchmark_results: dict[str, Any] | None = None,
    feature_importance: pd.Series | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "backtest_report.txt"

    lines = [
        "Backtest Report",
        "===============",
        "",
        "Summary Metrics",
        "---------------",
    ]
    for key, value in metrics.items():
        lines.append(_metric_line(key, value))

    if not snapshots.empty:
        latest = snapshots.iloc[-1]
        lines.extend(
            [
                "",
                "Latest Portfolio Snapshot",
                "------------------------",
                f"timestamp: {latest['timestamp']}",
                f"equity: {_format_value(float(latest['equity']))}",
                f"cash: {_format_value(float(latest['cash']))}",
                f"market_value: {_format_value(float(latest['market_value']))}",
                f"drawdown: {_format_percentage(float(latest['drawdown']))}",
                f"tax_liability: {_format_value(float(latest['tax_liability']))}",
            ]
        )

    lines.extend(
        [
            "",
            "Trade Summary",
            "-------------",
            f"trade_count: {len(trades)}",
            f"gross_notional: {_format_value(float(trades['gross_notional'].abs().sum())) if not trades.empty else '0.000000'}",
            f"fees_paid: {_format_value(float(trades['fees'].sum())) if not trades.empty else '0.000000'}",
            f"taxes_paid: {_format_value(float(trades['taxes_paid'].sum())) if not trades.empty else '0.000000'}",
        ]
    )

    if benchmark_results:
        lines.extend(["", "Benchmarks", "----------"])
        for name, result in benchmark_results.items():
            benchmark_return = result.metrics.get("total_return", 0.0)
            benchmark_sharpe = result.metrics.get("sharpe", 0.0)
            lines.append(
                f"{name}: total_return={_format_percentage(benchmark_return)}, sharpe={_format_value(benchmark_sharpe)}"
            )

    if feature_importance is not None and not feature_importance.empty:
        lines.extend(["", "Top Feature Importance", "----------------------"])
        for feature_name, importance in feature_importance.head(10).items():
            lines.append(f"{feature_name}: {_format_value(float(importance))}")

    lines.extend(
        [
            "",
            "Artifacts",
            "---------",
            f"equity_curve_png: {output_path / 'backtest_equity.png'}",
            f"drawdown_png: {output_path / 'backtest_drawdown.png'}",
            f"rolling_returns_png: {output_path / 'backtest_rolling_returns.png'}",
            f"metrics_csv: {output_path / 'backtest_metrics.csv'}",
            f"trades_csv: {output_path / 'trades.csv'}",
            f"signals_csv: {output_path / 'signals.csv'}",
            f"snapshots_csv: {output_path / 'snapshots.csv'}",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_walk_forward_report(output_dir: str | Path, summary: pd.DataFrame) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "walk_forward_report.txt"

    lines = [
        "Walk-Forward Report",
        "===================",
        "",
        f"fold_count: {len(summary)}",
    ]

    if not summary.empty:
        metric_columns = [
            "validation_metrics.total_return",
            "validation_metrics.sharpe",
            "test_metrics.total_return",
            "test_metrics.sharpe",
            "test_metrics.max_drawdown",
        ]
        present_columns = [column for column in metric_columns if column in summary.columns]

        lines.extend(["", "Aggregate Metrics", "-----------------"])
        for column in present_columns:
            series = pd.to_numeric(summary[column], errors="coerce").dropna()
            if series.empty:
                continue
            formatter = _format_percentage if "return" in column or "drawdown" in column else _format_value
            lines.append(f"{column}.mean: {formatter(float(series.mean()))}")
            lines.append(f"{column}.median: {formatter(float(series.median()))}")
            lines.append(f"{column}.min: {formatter(float(series.min()))}")
            lines.append(f"{column}.max: {formatter(float(series.max()))}")

        if "test_metrics.total_return" in summary.columns:
            ranked = summary.sort_values("test_metrics.total_return", ascending=False)
            best = ranked.iloc[0]
            worst = ranked.iloc[-1]
            lines.extend(
                [
                    "",
                    "Best Fold",
                    "---------",
                    f"fold_id: {best.get('fold_id')}",
                    f"test_total_return: {_format_percentage(float(best['test_metrics.total_return']))}",
                    f"test_sharpe: {_format_value(float(best.get('test_metrics.sharpe', 0.0)))}",
                    f"test_regime: {best.get('dominant_regimes.test', 'unknown')}",
                    "",
                    "Worst Fold",
                    "----------",
                    f"fold_id: {worst.get('fold_id')}",
                    f"test_total_return: {_format_percentage(float(worst['test_metrics.total_return']))}",
                    f"test_sharpe: {_format_value(float(worst.get('test_metrics.sharpe', 0.0)))}",
                    f"test_regime: {worst.get('dominant_regimes.test', 'unknown')}",
                ]
            )

        preview_columns = [column for column in ["fold_id", "dominant_regimes.train", "dominant_regimes.validation", "dominant_regimes.test", "test_metrics.total_return", "test_metrics.sharpe"] if column in summary.columns]
        if preview_columns:
            lines.extend(["", "Fold Preview", "------------"])
            preview = summary[preview_columns].copy()
            lines.append(preview.to_string(index=False))

    lines.extend(
        [
            "",
            "Artifacts",
            "---------",
            f"summary_csv: {output_path / 'walk_forward_summary.csv'}",
            f"summary_json: {output_path / 'walk_forward_summary.json'}",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_paper_report(output_dir: str | Path, results: list[dict[str, Any]]) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "paper_report.txt"

    lines = [
        "Paper Simulation Report",
        "=======================",
        "",
        f"decision_count: {len(results)}",
    ]

    if results:
        decisions = pd.DataFrame(results)
        lines.extend(
            [
                f"orders_submitted: {int((decisions['order_status'] != 'no_order').sum())}",
                f"mean_probability: {_format_value(float(decisions['predicted_probability'].mean()))}",
                f"mean_expected_return: {_format_percentage(float(decisions['expected_return'].mean()))}",
                "",
                "Decisions",
                "---------",
                decisions.to_string(index=False),
            ]
        )

    lines.extend(
        [
            "",
            "Artifacts",
            "---------",
            f"paper_json: {output_path / 'paper_simulation.json'}",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
