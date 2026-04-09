from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)


REQUIRED_COLUMNS = {"timestamp", "instrument", "open", "high", "low", "close", "volume"}


class MarketDataLoader:
    def __init__(self, timezone: str = "UTC") -> None:
        self.timezone = timezone
        self.logger = LOGGER

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        frame = self._normalize(frame)
        frame = self._handle_missing(frame)
        self.validate(frame, source=str(path))
        self.logger.info("Loaded %d market rows from %s", len(frame), path)
        return frame

    def validate(self, frame: pd.DataFrame, source: str = "dataframe") -> None:
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns from {source}: {sorted(missing)}")

        if frame.empty:
            raise ValueError("Market data is empty")

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for column in numeric_cols:
            if frame[column].isna().any():
                raise ValueError(f"Column {column!r} contains missing values in {source}")
            if (frame[column] <= 0).any():
                raise ValueError(f"Column {column!r} must be positive in {source}")

        if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
            raise ValueError("Detected high price below open/close")

        if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
            raise ValueError("Detected low price above open/close")

    def _normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        frame["timestamp"] = timestamps.dt.tz_convert(self.timezone).dt.tz_convert("UTC")
        frame["instrument"] = frame["instrument"].astype(str)
        frame["spread_bps"] = frame.get("spread_bps", pd.Series(5.0, index=frame.index)).fillna(5.0).astype(float)
        frame = frame.sort_values(["timestamp", "instrument"]).reset_index(drop=True)
        return frame

    def _handle_missing(self, frame: pd.DataFrame) -> pd.DataFrame:
        local = frame.copy()
        numeric_cols = ["open", "high", "low", "close"]
        local[numeric_cols] = local.groupby("instrument", sort=False)[numeric_cols].ffill().bfill()
        local["volume"] = local["volume"].fillna(0.0)
        local["spread_bps"] = local["spread_bps"].fillna(5.0)
        return local
