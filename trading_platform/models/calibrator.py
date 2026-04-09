from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from trading_platform.models.base import BaseSignalModel


class ProbabilityCalibrator(BaseSignalModel):
    def __init__(self, base_model: BaseSignalModel, method: str = "sigmoid") -> None:
        super().__init__(task="classification", random_state=base_model.random_state)
        self.base_model = base_model
        self.method = method

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProbabilityCalibrator":
        if self.base_model.estimator is None:
            self.base_model.fit(X, y)
        cv_folds = 3 if len(y) >= 30 else 2
        calibrated = CalibratedClassifierCV(self.base_model.estimator, method=self.method, cv=cv_folds)
        calibrated.fit(X, y.astype(int))
        self.estimator = calibrated
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise ValueError("Calibrator has not been fit")
        return self.estimator.predict(X)

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        return self.base_model.feature_importance(feature_names)
