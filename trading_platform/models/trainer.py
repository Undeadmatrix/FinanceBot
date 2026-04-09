from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from trading_platform.models.base import BaseSignalModel
from trading_platform.models.calibrator import ProbabilityCalibrator
from trading_platform.utils.validation import ModelConfig


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingResult:
    model: BaseSignalModel
    best_params: dict[str, object]


class ModelTrainer:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.logger = LOGGER

    def train(self, model: BaseSignalModel, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        best_params: dict[str, object] = {}
        if self.config.hyperparameter_search and getattr(model, "estimator", None) is None:
            model, best_params = self._search(model, X, y)
        else:
            model.fit(X, y)

        trained_model: BaseSignalModel = model
        if self.config.calibrate and model.task == "classification":
            trained_model = ProbabilityCalibrator(model).fit(X, y)

        self.logger.info("Model training complete")
        return TrainingResult(model=trained_model, best_params=best_params)

    def _search(self, model: BaseSignalModel, X: pd.DataFrame, y: pd.Series) -> tuple[BaseSignalModel, dict[str, object]]:
        if model.estimator is None:
            model.fit(X.iloc[: min(len(X), 10)], y.iloc[: min(len(y), 10)])
        estimator = model.estimator
        if estimator is None:
            return model.fit(X, y), {}

        grid = {}
        if hasattr(estimator, "named_steps") and "estimator" in estimator.named_steps:
            inner = estimator.named_steps["estimator"]
            name = inner.__class__.__name__.lower()
            if "logistic" in name:
                grid = {"estimator__C": [0.25, 1.0, 4.0]}
            elif "randomforest" in name:
                grid = {"estimator__max_depth": [3, 5], "estimator__min_samples_leaf": [5, 10]}

        if not grid:
            return model.fit(X, y), {}

        splitter = TimeSeriesSplit(n_splits=self.config.cv_splits)
        search = GridSearchCV(estimator, grid, cv=splitter, n_jobs=1)
        search.fit(X, y)
        model.estimator = search.best_estimator_
        LOGGER.info("Best parameters found: %s", search.best_params_)
        return model, dict(search.best_params_)
