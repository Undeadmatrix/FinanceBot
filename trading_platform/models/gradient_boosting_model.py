from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from trading_platform.models.base import BaseSignalModel


class GradientBoostingModel(BaseSignalModel):
    def __init__(self, task: str = "classification", random_state: int = 7) -> None:
        super().__init__(task=task, random_state=random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingModel":
        estimator = (
            GradientBoostingClassifier(random_state=self.random_state, max_depth=3)
            if self.task == "classification"
            else GradientBoostingRegressor(random_state=self.random_state, max_depth=3)
        )
        pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("estimator", estimator)])
        pipeline.fit(X, y)
        self.estimator = pipeline
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise ValueError("Model has not been fit")
        return self.estimator.predict(X)


def optional_gradient_boosting_library() -> str | None:
    if importlib.util.find_spec("xgboost") is not None:
        return "xgboost"
    if importlib.util.find_spec("lightgbm") is not None:
        return "lightgbm"
    return None
