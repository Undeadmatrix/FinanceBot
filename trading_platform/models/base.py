from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class BaseSignalModel(ABC):
    def __init__(self, task: str, random_state: int = 7) -> None:
        self.task = task
        self.random_state = random_state
        self.estimator: Any | None = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseSignalModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise ValueError("Model has not been fit")
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            return proba.ravel()
        predictions = self.predict(X)
        return np.asarray(predictions, dtype=float)

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        if self.estimator is None:
            raise ValueError("Model has not been fit")

        estimator = self.estimator
        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
        elif hasattr(estimator, "named_steps") and "estimator" in estimator.named_steps:
            nested = estimator.named_steps["estimator"]
            if hasattr(nested, "feature_importances_"):
                values = nested.feature_importances_
            elif hasattr(nested, "coef_"):
                values = np.ravel(np.abs(nested.coef_))
            else:
                values = np.zeros(len(feature_names), dtype=float)
        elif hasattr(estimator, "coef_"):
            values = np.ravel(np.abs(estimator.coef_))
        else:
            values = np.zeros(len(feature_names), dtype=float)
        return pd.Series(values, index=feature_names).sort_values(ascending=False)

    def save(self, path: str | Path) -> Path:
        if self.estimator is None:
            raise ValueError("Model has not been fit")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "BaseSignalModel":
        model = joblib.load(path)
        if not isinstance(model, BaseSignalModel):
            raise TypeError(f"Unexpected object loaded from {path}: {type(model)!r}")
        return model
