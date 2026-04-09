from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


class TradeLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists() and self.path.stat().st_size > 0

    def append(self, rows: Iterable[dict[str, object]]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerows(rows)
