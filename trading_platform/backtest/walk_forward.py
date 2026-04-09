from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trading_platform.backtest.engine import BacktestEngine, BacktestResult
from trading_platform.data.dataset_builder import DatasetBundle
from trading_platform.utils.serialization import dump_json, model_to_dict
from trading_platform.utils.validation import PlatformConfig


@dataclass(slots=True)
class WalkForwardFoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    dominant_regimes: dict[str, str]


class WalkForwardValidator:
    def __init__(self, config: PlatformConfig) -> None:
        self.config = config

    def run(self, bundle: DatasetBundle, output_dir: str | Path | None = None) -> pd.DataFrame:
        dataset = bundle.dataset.sort_values(["timestamp", "instrument"]).reset_index(drop=True)
        dates = sorted(dataset["timestamp"].unique())
        wf = self.config.walk_forward
        rows: list[dict[str, Any]] = []

        fold_id = 0
        cursor = 0
        while cursor + wf.train_bars + wf.validation_bars + wf.test_bars <= len(dates):
            train_dates = dates[cursor : cursor + wf.train_bars]
            validation_dates = dates[cursor + wf.train_bars : cursor + wf.train_bars + wf.validation_bars]
            test_dates = dates[
                cursor + wf.train_bars + wf.validation_bars : cursor + wf.train_bars + wf.validation_bars + wf.test_bars
            ]

            validation_bundle = DatasetBundle(
                raw=bundle.raw,
                dataset=dataset[dataset["timestamp"].isin([*train_dates, *validation_dates])].copy(),
                feature_columns=bundle.feature_columns,
                target_column=bundle.target_column,
            )
            validation_result = self._run_fold(
                validation_bundle,
                min_train_bars=len(train_dates),
                include_benchmarks=False,
            )

            test_bundle = DatasetBundle(
                raw=bundle.raw,
                dataset=dataset[dataset["timestamp"].isin([*train_dates, *validation_dates, *test_dates])].copy(),
                feature_columns=bundle.feature_columns,
                target_column=bundle.target_column,
            )
            test_result = self._run_fold(
                test_bundle,
                min_train_bars=len(train_dates) + len(validation_dates),
                include_benchmarks=False,
            )

            dominant_regimes = self._dominant_regimes(
                dataset,
                train_dates=train_dates,
                validation_dates=validation_dates,
                test_dates=test_dates,
            )
            rows.append(
                model_to_dict(
                    WalkForwardFoldResult(
                        fold_id=fold_id,
                        train_start=pd.Timestamp(train_dates[0]),
                        train_end=pd.Timestamp(train_dates[-1]),
                        validation_start=pd.Timestamp(validation_dates[0]),
                        validation_end=pd.Timestamp(validation_dates[-1]),
                        test_start=pd.Timestamp(test_dates[0]),
                        test_end=pd.Timestamp(test_dates[-1]),
                        validation_metrics=validation_result.metrics,
                        test_metrics=test_result.metrics,
                        dominant_regimes=dominant_regimes,
                    )
                )
            )
            fold_id += 1
            cursor += wf.step_bars

        summary = pd.json_normalize(rows, sep=".")
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            summary.to_csv(output_path / "walk_forward_summary.csv", index=False)
            dump_json(output_path / "walk_forward_summary.json", rows)
        return summary

    def _run_fold(self, bundle: DatasetBundle, min_train_bars: int, include_benchmarks: bool) -> BacktestResult:
        payload = model_to_dict(self.config.backtest)
        payload["min_train_bars"] = min_train_bars
        payload["benchmark"] = [] if not include_benchmarks else payload.get("benchmark", [])
        engine = BacktestEngine(
            backtest_config=type(self.config.backtest)(**payload),
            execution_config=self.config.execution,
            tax_config=self.config.tax,
            risk_config=self.config.risk,
            broker_config=self.config.broker,
            strategy_config=self.config.strategy,
            model_config=self.config.model,
        )
        return engine.run(bundle, include_benchmarks=include_benchmarks)

    def _dominant_regimes(
        self,
        dataset: pd.DataFrame,
        train_dates: list[pd.Timestamp],
        validation_dates: list[pd.Timestamp],
        test_dates: list[pd.Timestamp],
    ) -> dict[str, str]:
        if "regime" not in dataset.columns:
            return {"train": "unknown", "validation": "unknown", "test": "unknown"}

        def mode_for(dates_subset: list[pd.Timestamp]) -> str:
            subset = dataset[dataset["timestamp"].isin(dates_subset)]
            if subset.empty:
                return "unknown"
            return str(subset["regime"].mode().iloc[0])

        return {
            "train": mode_for(train_dates),
            "validation": mode_for(validation_dates),
            "test": mode_for(test_dates),
        }
