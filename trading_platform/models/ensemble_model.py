from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from trading_platform.models.base import BaseSignalModel


class EnsembleModel(BaseSignalModel):
    def __init__(self, models: Sequence[BaseSignalModel], task: str = "classification", random_state: int = 7) -> None:
        super().__init__(task=task, random_state=random_state)
        self.models = list(models)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel":
        for model in self.models:
            model.fit(X, y)
        self.estimator = self
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "classification":
            return (self.predict_proba(X) >= 0.5).astype(int)
        stacked = np.vstack([model.predict(X) for model in self.models])
        return stacked.mean(axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise ValueError("Ensemble contains no models")
        stacked = np.vstack([model.predict_proba(X) for model in self.models])
        return stacked.mean(axis=0)
