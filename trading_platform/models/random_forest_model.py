from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from trading_platform.models.base import BaseSignalModel


class RandomForestModel(BaseSignalModel):
    def __init__(self, task: str = "classification", random_state: int = 7) -> None:
        super().__init__(task=task, random_state=random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        estimator = (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=10,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
            if self.task == "classification"
            else RandomForestRegressor(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=10,
                random_state=self.random_state,
                n_jobs=-1,
            )
        )
        pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("estimator", estimator)])
        pipeline.fit(X, y)
        self.estimator = pipeline
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise ValueError("Model has not been fit")
        return self.estimator.predict(X)
