from __future__ import annotations

from datetime import timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_utc_index(values: Iterable[object], tz: str = "UTC") -> pd.DatetimeIndex:
    """Convert timestamps to a UTC-normalized DatetimeIndex."""
    index = pd.DatetimeIndex(pd.to_datetime(list(values), utc=True))
    if tz and tz.upper() != "UTC":
        return index.tz_convert(tz).tz_convert("UTC")
    return index


def ensure_utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    else:
        timestamp = timestamp.tz_convert(timezone.utc)
    return timestamp


def floor_to_day(value: object) -> pd.Timestamp:
    return ensure_utc_timestamp(value).floor("D")


def make_output_dir(base_dir: str | Path, run_id: str) -> Path:
    path = Path(base_dir) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path
