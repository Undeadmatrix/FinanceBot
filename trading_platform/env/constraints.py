from __future__ import annotations


def clip_fraction(value: float, max_abs_fraction: float) -> float:
    return max(-max_abs_fraction, min(max_abs_fraction, value))
