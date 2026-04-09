from __future__ import annotations

from trading_platform.backtest.engine import BacktestEngine, BacktestResult
from trading_platform.data.dataset_builder import DatasetBundle


class BenchmarkSuite:
    def __init__(self, engine: BacktestEngine) -> None:
        self.engine = engine

    def run(self, bundle: DatasetBundle) -> dict[str, BacktestResult]:
        return {
            benchmark_name: self.engine.run_benchmark_strategy(bundle, benchmark_name)
            for benchmark_name in self.engine.backtest_config.benchmark
        }
