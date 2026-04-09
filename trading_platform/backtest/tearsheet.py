from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from trading_platform.backtest.metrics import compute_performance_metrics
from trading_platform.utils.serialization import dump_json


class TearSheetGenerator:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        snapshots: pd.DataFrame,
        trades: pd.DataFrame,
        benchmark_snapshots: pd.DataFrame | None = None,
        prefix: str = "tearsheet",
    ) -> dict[str, Any]:
        metrics = compute_performance_metrics(snapshots, trades, benchmark_snapshots=benchmark_snapshots)
        self._plot_equity(snapshots, benchmark_snapshots, prefix)
        self._plot_drawdown(snapshots, prefix)
        self._plot_rolling_returns(snapshots, prefix)
        dump_json(self.output_dir / f"{prefix}_metrics.json", metrics)
        pd.DataFrame([metrics]).to_csv(self.output_dir / f"{prefix}_metrics.csv", index=False)
        return metrics

    def _plot_equity(self, snapshots: pd.DataFrame, benchmark_snapshots: pd.DataFrame | None, prefix: str) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(snapshots["timestamp"], snapshots["equity"], label="Strategy")
        if benchmark_snapshots is not None and not benchmark_snapshots.empty:
            plt.plot(benchmark_snapshots["timestamp"], benchmark_snapshots["equity"], label="Benchmark", alpha=0.75)
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}_equity.png", dpi=150)
        plt.close()

    def _plot_drawdown(self, snapshots: pd.DataFrame, prefix: str) -> None:
        plt.figure(figsize=(10, 4))
        plt.plot(snapshots["timestamp"], snapshots["drawdown"], color="firebrick")
        plt.title("Drawdown")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}_drawdown.png", dpi=150)
        plt.close()

    def _plot_rolling_returns(self, snapshots: pd.DataFrame, prefix: str) -> None:
        rolling = snapshots["equity"].pct_change().rolling(21, min_periods=5).sum()
        plt.figure(figsize=(10, 4))
        plt.plot(snapshots["timestamp"], rolling, color="navy")
        plt.title("Rolling 21-Bar Returns")
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}_rolling_returns.png", dpi=150)
        plt.close()
