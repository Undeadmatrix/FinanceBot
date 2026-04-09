from __future__ import annotations

from trading_platform.models.base import BaseSignalModel
from trading_platform.models.ensemble_model import EnsembleModel
from trading_platform.models.gradient_boosting_model import GradientBoostingModel
from trading_platform.models.logistic_model import LogisticRegressionModel
from trading_platform.models.random_forest_model import RandomForestModel
from trading_platform.utils.validation import ModelConfig


def build_model(config: ModelConfig) -> BaseSignalModel:
    if config.name == "logistic_regression":
        return LogisticRegressionModel(max_iter=config.max_iter, random_state=config.random_state)
    if config.name == "random_forest":
        return RandomForestModel(task=config.task, random_state=config.random_state)
    if config.name == "gradient_boosting":
        return GradientBoostingModel(task=config.task, random_state=config.random_state)
    if config.name == "ensemble":
        models = [
            LogisticRegressionModel(max_iter=config.max_iter, random_state=config.random_state),
            RandomForestModel(task=config.task, random_state=config.random_state),
        ]
        return EnsembleModel(models=models, task=config.task, random_state=config.random_state)
    raise ValueError(f"Unsupported model name: {config.name}")
