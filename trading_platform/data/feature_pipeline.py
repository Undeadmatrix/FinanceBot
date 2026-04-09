from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from trading_platform.utils.validation import FeatureConfig


LOGGER = logging.getLogger(__name__)


class FeaturePipeline:
    """Leakage-safe feature engineering using only current and past bars."""

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.logger = LOGGER

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"timestamp", "instrument", "open", "high", "low", "close", "volume"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns for features: {sorted(missing)}")

        engineered = []
        for instrument, group in frame.groupby("instrument", sort=False):
            engineered.append(self._transform_single(group.sort_values("timestamp").reset_index(drop=True)))
        combined = pd.concat(engineered, ignore_index=True)
        combined = self._add_cross_sectional_features(combined)
        self.logger.info("Built %d feature columns", len(self.feature_columns(combined)))
        return combined

    def feature_columns(self, frame: pd.DataFrame) -> list[str]:
        base_cols = {
            "timestamp",
            "instrument",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "spread_bps",
            "regime",
            "liquidity_score",
        }
        label_prefixes = ("forward_return_", "target")
        label_columns = {"next_period_direction", "expected_edge_after_cost", "threshold_action"}
        return [
            column
            for column in frame.columns
            if column not in base_cols and column not in label_columns and not column.startswith(label_prefixes)
        ]

    def _transform_single(self, group: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        local = group.copy()
        close = local["close"]
        volume = local["volume"]
        returns = close.pct_change()

        local["return_1"] = returns
        for lag in cfg.lags:
            local[f"return_lag_{lag}"] = returns.shift(lag)

        for window in cfg.rolling_windows:
            local[f"rolling_return_{window}"] = close.pct_change(window)
            local[f"rolling_vol_{window}"] = returns.rolling(window, min_periods=max(2, window // 2)).std()
            local[f"ma_ratio_{window}"] = close / close.rolling(window, min_periods=max(2, window // 2)).mean() - 1.0
            local[f"volume_delta_{window}"] = volume / volume.rolling(window, min_periods=max(2, window // 2)).mean() - 1.0
            rolling_mean = close.rolling(window, min_periods=max(2, window // 2)).mean()
            rolling_std = close.rolling(window, min_periods=max(2, window // 2)).std()
            local[f"zscore_{window}"] = (close - rolling_mean) / rolling_std.replace(0.0, np.nan)
            rolling_high = local["high"].rolling(window, min_periods=max(2, window // 2)).max()
            rolling_low = local["low"].rolling(window, min_periods=max(2, window // 2)).min()
            local[f"distance_to_high_{window}"] = close / rolling_high - 1.0
            local[f"distance_to_low_{window}"] = close / rolling_low - 1.0

        local["realized_vol_proxy"] = np.log1p((local["high"] - local["low"]) / local["open"]).rolling(
            cfg.volatility_window,
            min_periods=max(2, cfg.volatility_window // 2),
        ).mean()

        delta = close.diff()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)
        avg_gain = gains.rolling(cfg.rsi_window, min_periods=max(2, cfg.rsi_window // 2)).mean()
        avg_loss = losses.rolling(cfg.rsi_window, min_periods=max(2, cfg.rsi_window // 2)).mean().replace(0.0, np.nan)
        rs = avg_gain / avg_loss
        local["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        ema_fast = close.ewm(span=cfg.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=cfg.macd_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=cfg.macd_signal, adjust=False).mean()
        local["macd"] = macd
        local["macd_signal"] = macd_signal
        local["macd_hist"] = macd - macd_signal

        rolling_peak = close.rolling(cfg.drawdown_window, min_periods=1).max()
        local["drawdown_state"] = close / rolling_peak - 1.0
        local["trend_strength"] = ema_fast / ema_slow - 1.0
        local["vol_regime"] = local[f"rolling_vol_{cfg.volatility_window}"] / (
            local[f"rolling_vol_{cfg.rolling_windows[-1]}"].replace(0.0, np.nan)
        )
        local["volume_shock"] = volume.pct_change().rolling(5, min_periods=2).mean()
        local["bar_range"] = (local["high"] - local["low"]) / local["open"]

        return local

    def _add_cross_sectional_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        local = frame.copy()
        vol_column = f"rolling_vol_{self.config.volatility_window}"
        if vol_column not in local.columns:
            vol_candidates = [column for column in local.columns if column.startswith("rolling_vol_")]
            vol_column = vol_candidates[0] if vol_candidates else "return_1"
        local["xs_return_rank_1"] = local.groupby("timestamp")["return_1"].rank(pct=True)
        local["xs_volume_rank"] = local.groupby("timestamp")["volume"].rank(pct=True)
        local["xs_vol_rank"] = local.groupby("timestamp")[vol_column].rank(pct=True)
        return local
