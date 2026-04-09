from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trading_platform.models.base import BaseSignalModel


class LogisticRegressionModel(BaseSignalModel):
    def __init__(self, max_iter: int = 1000, random_state: int = 7) -> None:
        super().__init__(task="classification", random_state=random_state)
        self.max_iter = max_iter

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionModel":
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "estimator",
                    LogisticRegression(
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        estimator.fit(X, y.astype(int))
        self.estimator = estimator
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise ValueError("Model has not been fit")
        return self.estimator.predict(X)
