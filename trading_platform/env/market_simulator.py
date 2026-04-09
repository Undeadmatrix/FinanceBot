from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class MarketBar:
    timestamp: pd.Timestamp
    instrument: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread_bps: float


class MarketSimulator:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.sort_values(["timestamp", "instrument"]).reset_index(drop=True)

    def iter_dates(self) -> list[pd.Timestamp]:
        return list(pd.Index(self.frame["timestamp"].drop_duplicates()).sort_values())

    def bars_for_date(self, timestamp: pd.Timestamp) -> pd.DataFrame:
        return self.frame[self.frame["timestamp"] == timestamp].copy()
