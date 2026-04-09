from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from trading_platform.utils.validation import SyntheticDataConfig


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RegimeSpec:
    name: str
    drift: float
    autoregressive: float
    mean_reversion: float
    vol_multiplier: float
    spread_multiplier: float
    volume_multiplier: float
    jump_bias: float = 0.0


REGIMES: tuple[RegimeSpec, ...] = (
    RegimeSpec("trend_up", 0.0012, 0.25, 0.02, 0.9, 0.9, 1.0, 0.0),
    RegimeSpec("trend_down", -0.0010, 0.20, 0.02, 1.0, 1.0, 0.95, 0.0),
    RegimeSpec("mean_revert", 0.0001, -0.15, 0.18, 0.8, 0.8, 0.85, 0.0),
    RegimeSpec("high_vol", 0.0002, 0.05, 0.05, 1.8, 1.8, 1.5, 0.0),
    RegimeSpec("crash", -0.0030, 0.05, 0.02, 2.6, 2.5, 2.2, -0.015),
)


class SyntheticMarketGenerator:
    """Generate multi-asset OHLCV data with regime shifts and microstructure frictions."""

    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config
        self.logger = LOGGER

    def generate(self) -> pd.DataFrame:
        frames = [self._generate_instrument(symbol, seed_offset=i) for i, symbol in enumerate(self.config.instruments)]
        data = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "instrument"]).reset_index(drop=True)
        self.logger.info("Generated synthetic data for %d instruments and %d rows", len(self.config.instruments), len(data))
        return data

    def _generate_instrument(self, instrument: str, seed_offset: int) -> pd.DataFrame:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed + seed_offset * 97)
        dates = pd.date_range(cfg.start_date, periods=cfg.periods, freq=cfg.freq, tz="UTC")

        closes = np.zeros(cfg.periods, dtype=float)
        opens = np.zeros(cfg.periods, dtype=float)
        highs = np.zeros(cfg.periods, dtype=float)
        lows = np.zeros(cfg.periods, dtype=float)
        volumes = np.zeros(cfg.periods, dtype=float)
        spreads = np.zeros(cfg.periods, dtype=float)
        returns = np.zeros(cfg.periods, dtype=float)
        current_regime = REGIMES[seed_offset % len(REGIMES)]
        regimes: list[str] = [current_regime.name]

        closes[0] = cfg.start_price * (1 + 0.05 * seed_offset)
        opens[0] = closes[0]
        highs[0] = closes[0] * 1.003
        lows[0] = closes[0] * 0.997
        volumes[0] = 1_000_000.0
        spreads[0] = 6.0
        anchor_price = closes[0]

        for idx in range(1, cfg.periods):
            if rng.random() < cfg.regime_shift_probability:
                current_regime = REGIMES[int(rng.integers(0, len(REGIMES)))]

            regimes.append(current_regime.name)
            previous_close = closes[idx - 1]
            previous_return = returns[idx - 1]

            anchor_price = 0.995 * anchor_price + 0.005 * previous_close
            distance_from_anchor = np.log(max(previous_close, 1e-8) / max(anchor_price, 1e-8))

            clustered_vol = cfg.volatility * (
                1.0
                + cfg.volatility_cluster * abs(previous_return) * 50.0
                + 0.25 * rng.random()
            )
            state_vol = clustered_vol * current_regime.vol_multiplier

            overnight_gap = rng.normal(loc=0.0, scale=state_vol * 0.35)
            opens[idx] = max(previous_close * np.exp(overnight_gap), 1.0)

            drift = cfg.trend_drift + current_regime.drift
            autoregressive = current_regime.autoregressive * previous_return
            mean_reversion = -(
                cfg.mean_reversion_strength + current_regime.mean_reversion
            ) * distance_from_anchor
            innovation = rng.normal(loc=0.0, scale=state_vol)

            jump = 0.0
            if rng.random() < cfg.jump_probability:
                jump = rng.normal(current_regime.jump_bias, cfg.jump_scale)

            outlier = 0.0
            if rng.random() < cfg.outlier_probability:
                outlier = rng.normal(0.0, cfg.jump_scale * 1.5)

            intraday_return = drift + autoregressive + mean_reversion + innovation + jump + outlier
            close_price = max(opens[idx] * np.exp(intraday_return), 1.0)

            bar_range = abs(intraday_return) + abs(rng.normal(0.0, state_vol * 0.75))
            high_price = max(opens[idx], close_price) * (1.0 + bar_range * 0.6)
            low_price = min(opens[idx], close_price) * max(0.2, 1.0 - bar_range * 0.6)

            returns[idx] = close_price / previous_close - 1.0
            closes[idx] = close_price
            highs[idx] = max(high_price, opens[idx], closes[idx])
            lows[idx] = min(low_price, opens[idx], closes[idx])

            volume_shock = 1.0 + abs(intraday_return) * 8.0 + abs(jump) * 12.0 + rng.lognormal(0.0, 0.15)
            volumes[idx] = max(100_000.0, 850_000.0 * current_regime.volume_multiplier * volume_shock)

            spreads[idx] = max(
                1.0,
                4.5
                * current_regime.spread_multiplier
                * (1.0 + abs(intraday_return) * 20.0 + 0.25 * rng.random()),
            )

        frame = pd.DataFrame(
            {
                "timestamp": dates,
                "instrument": instrument,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
                "spread_bps": spreads,
                "regime": regimes[: cfg.periods],
            }
        )
        frame["liquidity_score"] = np.clip(frame["volume"] / frame["volume"].rolling(20, min_periods=1).median(), 0.5, 3.0)
        return frame


def generate_synthetic_market(config: SyntheticDataConfig | dict[str, object]) -> pd.DataFrame:
    parsed = config if isinstance(config, SyntheticDataConfig) else SyntheticDataConfig(**config)
    return SyntheticMarketGenerator(parsed).generate()
