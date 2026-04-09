from __future__ import annotations

import pandas as pd

from trading_platform.utils.validation import ExecutionConfig, LabelConfig


class LabelBuilder:
    """Construct leakage-safe targets aligned to next-bar execution."""

    def __init__(self, config: LabelConfig, execution_config: ExecutionConfig) -> None:
        self.config = config
        self.execution_config = execution_config

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        labeled = []
        for _, group in frame.groupby("instrument", sort=False):
            labeled.append(self._label_single(group.sort_values("timestamp").reset_index(drop=True)))
        return pd.concat(labeled, ignore_index=True)

    def _label_single(self, group: pd.DataFrame) -> pd.DataFrame:
        local = group.copy()
        cost_rate = self._estimated_cost_rate(local)

        next_open = local["open"].shift(-1)
        for horizon in sorted(set([self.config.horizon, *self.config.horizons])):
            future_close = local["close"].shift(-horizon)
            local[f"forward_return_{horizon}"] = future_close / next_open - 1.0

        primary_forward = local[f"forward_return_{self.config.horizon}"]
        local["next_period_direction"] = (primary_forward > 0.0).astype(float)
        local["expected_edge_after_cost"] = primary_forward - cost_rate - self.config.cost_buffer_bps / 10_000.0
        local["threshold_action"] = 0
        local.loc[local["expected_edge_after_cost"] > self.config.action_threshold, "threshold_action"] = 1
        local.loc[local["expected_edge_after_cost"] < -self.config.action_threshold, "threshold_action"] = -1
        invalid_mask = primary_forward.isna()
        local.loc[invalid_mask, ["next_period_direction", "expected_edge_after_cost", "threshold_action"]] = pd.NA

        target_map = {
            "next_direction": "next_period_direction",
            "next_return": f"forward_return_{self.config.horizon}",
            "multi_horizon_return": f"forward_return_{self.config.horizon}",
            "edge_after_cost": "expected_edge_after_cost",
            "threshold_action": "threshold_action",
        }
        local["target"] = local[target_map[self.config.target]]
        return local

    def _estimated_cost_rate(self, frame: pd.DataFrame) -> pd.Series:
        total_bps = (
            frame.get("spread_bps", pd.Series(self.execution_config.spread_bps, index=frame.index)).fillna(
                self.execution_config.spread_bps
            )
            + self.execution_config.slippage_bps
            + self.execution_config.fee_bps
            + self.execution_config.market_impact_bps
        )
        return total_bps / 10_000.0
