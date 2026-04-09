from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_platform.utils.math_utils import annualize_return, annualize_volatility, downside_deviation, rolling_drawdown, safe_divide


def compute_performance_metrics(
    snapshots: pd.DataFrame,
    trades: pd.DataFrame,
    benchmark_snapshots: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if snapshots.empty:
        return {}

    equity = snapshots["equity"].astype(float)
    returns = equity.pct_change().fillna(0.0)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    periods = max(len(returns) - 1, 1)
    annual_return = annualize_return(total_return, periods)
    annual_vol = annualize_volatility(returns)
    downside = downside_deviation(returns)
    sharpe = safe_divide(returns.mean() * 252.0, annual_vol)
    sortino = safe_divide(returns.mean() * 252.0, downside)
    max_drawdown = float(rolling_drawdown(equity).min())
    calmar = safe_divide(annual_return, abs(max_drawdown))

    wins = trades[trades["realized_pnl"] > 0]
    losses = trades[trades["realized_pnl"] < 0]
    profit_factor = safe_divide(wins["realized_pnl"].sum(), abs(losses["realized_pnl"].sum()))

    metrics: dict[str, Any] = {
        "total_return": float(total_return),
        "cagr": float(annual_return),
        "volatility": float(annual_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "win_rate": float(len(wins) / len(trades)) if len(trades) else 0.0,
        "average_win": float(wins["realized_pnl"].mean()) if not wins.empty else 0.0,
        "average_loss": float(losses["realized_pnl"].mean()) if not losses.empty else 0.0,
        "profit_factor": float(profit_factor),
        "turnover": float(snapshots["turnover"].iloc[-1]),
        "exposure": float(snapshots["gross_exposure"].mean()),
        "fee_drag": float(snapshots["fees_paid"].iloc[-1] / equity.iloc[0]),
        "tax_drag": float((snapshots["taxes_paid"].iloc[-1] + snapshots["tax_liability"].iloc[-1]) / equity.iloc[0]),
        "trade_count": int(len(trades)),
    }

    if benchmark_snapshots is not None and not benchmark_snapshots.empty:
        benchmark_equity = benchmark_snapshots["equity"].astype(float)
        benchmark_total_return = benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1.0
        benchmark_returns = benchmark_equity.pct_change().fillna(0.0)
        active_return = returns - benchmark_returns.reindex(returns.index, fill_value=0.0)
        tracking_error = annualize_volatility(active_return)
        metrics["benchmark_total_return"] = float(benchmark_total_return)
        metrics["excess_total_return"] = float(total_return - benchmark_total_return)
        metrics["information_ratio"] = float(safe_divide(active_return.mean() * 252.0, tracking_error))

    return metrics
