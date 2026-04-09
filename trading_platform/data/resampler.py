from __future__ import annotations

import pandas as pd


def resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV per instrument with extension hooks for intraday support."""
    required = {"timestamp", "instrument", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns for resampling: {sorted(missing)}")

    results = []
    for instrument, group in frame.groupby("instrument", sort=False):
        local = group.set_index("timestamp").sort_index()
        resampled = local.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "spread_bps": "mean",
            }
        )
        resampled["instrument"] = instrument
        resampled = resampled.dropna(subset=["open", "high", "low", "close"]).reset_index()
        results.append(resampled)
    return pd.concat(results, ignore_index=True).sort_values(["timestamp", "instrument"]).reset_index(drop=True)
