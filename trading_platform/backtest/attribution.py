from __future__ import annotations

from typing import Any

import pandas as pd


def build_cost_tax_attribution(trades: pd.DataFrame, snapshots: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "gross_notional": 0.0,
            "fees_paid": 0.0,
            "taxes_paid": 0.0,
            "tax_liability": float(snapshots["tax_liability"].iloc[-1]) if not snapshots.empty else 0.0,
            "net_realized_pnl": 0.0,
        }

    return {
        "gross_notional": float(trades["gross_notional"].abs().sum()),
        "fees_paid": float(trades["fees"].sum()),
        "taxes_paid": float(trades["taxes_paid"].sum()),
        "tax_liability": float(snapshots["tax_liability"].iloc[-1]) if not snapshots.empty else 0.0,
        "net_realized_pnl": float(trades["realized_pnl"].sum() - trades["fees"].sum() - trades["taxes_paid"].sum()),
    }
