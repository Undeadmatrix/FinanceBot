from __future__ import annotations

import pandas as pd

from trading_platform.data.feature_pipeline import FeaturePipeline
from trading_platform.data.labeler import LabelBuilder
from trading_platform.data.synthetic_generator import generate_synthetic_market
from trading_platform.tests.helpers import make_test_config


def test_feature_pipeline_is_future_stable(tmp_path):
    config = make_test_config(tmp_path)
    config.market_data.synthetic.periods = 120
    frame = generate_synthetic_market(config.market_data.synthetic)
    instrument_frame = frame[frame["instrument"] == "TEST_A"].reset_index(drop=True)

    pipeline = FeaturePipeline(config.features)
    full_features = pipeline.transform(instrument_frame)
    truncated_features = pipeline.transform(instrument_frame.iloc[:80].copy())

    comparison_index = 60
    feature_columns = pipeline.feature_columns(full_features)
    pd.testing.assert_series_equal(
        full_features.loc[comparison_index, feature_columns],
        truncated_features.loc[comparison_index, feature_columns],
        check_names=False,
    )


def test_label_builder_drops_unresolved_tail_rows(tmp_path):
    config = make_test_config(tmp_path)
    config.market_data.synthetic.periods = 80
    frame = generate_synthetic_market(config.market_data.synthetic)
    instrument_frame = frame[frame["instrument"] == "TEST_A"].reset_index(drop=True)

    features = FeaturePipeline(config.features).transform(instrument_frame)
    labeled = LabelBuilder(config.labels, config.execution).transform(features)
    assert labeled["target"].iloc[-1] is pd.NA or pd.isna(labeled["target"].iloc[-1])
