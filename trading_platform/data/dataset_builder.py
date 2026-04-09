from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_platform.data.feature_pipeline import FeaturePipeline
from trading_platform.data.labeler import LabelBuilder


@dataclass(slots=True)
class DatasetBundle:
    raw: pd.DataFrame
    dataset: pd.DataFrame
    feature_columns: list[str]
    target_column: str


class DatasetBuilder:
    def __init__(self, feature_pipeline: FeaturePipeline, label_builder: LabelBuilder) -> None:
        self.feature_pipeline = feature_pipeline
        self.label_builder = label_builder

    def build(self, frame: pd.DataFrame) -> DatasetBundle:
        features = self.feature_pipeline.transform(frame)
        labeled = self.label_builder.transform(features)
        feature_columns = self.feature_pipeline.feature_columns(labeled)
        dataset = labeled.dropna(subset=feature_columns + ["target"]).sort_values(["timestamp", "instrument"]).reset_index(
            drop=True
        )
        return DatasetBundle(raw=frame, dataset=dataset, feature_columns=feature_columns, target_column="target")

    def time_split(
        self,
        dataset: pd.DataFrame,
        train_fraction: float = 0.6,
        validation_fraction: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")

        timestamps = sorted(dataset["timestamp"].unique())
        train_end = int(len(timestamps) * train_fraction)
        valid_end = train_end + int(len(timestamps) * validation_fraction)

        train_dates = set(timestamps[:train_end])
        valid_dates = set(timestamps[train_end:valid_end])
        test_dates = set(timestamps[valid_end:])

        return (
            dataset[dataset["timestamp"].isin(train_dates)].copy(),
            dataset[dataset["timestamp"].isin(valid_dates)].copy(),
            dataset[dataset["timestamp"].isin(test_dates)].copy(),
        )
