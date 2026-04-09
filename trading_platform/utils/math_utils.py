from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or math.isclose(denominator, 0.0):
        return default
    return numerator / denominator


def annualize_return(total_return: float, periods: int, periods_per_year: int = 252) -> float:
    if periods <= 0:
        return 0.0
    if 1 + total_return <= 0:
        return -1.0
    return (1 + total_return) ** (periods_per_year / periods) - 1


def annualize_volatility(returns: Iterable[float], periods_per_year: int = 252) -> float:
    array = np.asarray(list(returns), dtype=float)
    if array.size <= 1:
        return 0.0
    return float(np.nanstd(array, ddof=1) * math.sqrt(periods_per_year))


def rolling_drawdown(equity: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(equity), dtype=float)
    if array.size == 0:
        return np.array([], dtype=float)
    peaks = np.maximum.accumulate(array)
    return np.where(peaks == 0, 0.0, array / peaks - 1.0)


def downside_deviation(returns: Iterable[float], periods_per_year: int = 252) -> float:
    array = np.asarray(list(returns), dtype=float)
    downside = np.minimum(array, 0.0)
    if downside.size <= 1:
        return 0.0
    return float(np.sqrt(np.mean(np.square(downside))) * math.sqrt(periods_per_year))
