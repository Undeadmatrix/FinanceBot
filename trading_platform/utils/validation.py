from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SyntheticDataConfig(BaseModel):
    instruments: list[str] = Field(default_factory=lambda: ["SYNTH_A"])
    start_date: str = "2020-01-01"
    periods: int = 800
    freq: str = "B"
    seed: int = 7
    start_price: float = 100.0
    trend_drift: float = 0.0008
    mean_reversion_strength: float = 0.15
    volatility: float = 0.015
    volatility_cluster: float = 0.25
    jump_probability: float = 0.02
    jump_scale: float = 0.05
    regime_shift_probability: float = 0.05
    outlier_probability: float = 0.01


class MarketDataConfig(BaseModel):
    source: Literal["synthetic", "csv"] = "synthetic"
    csv_path: str | None = None
    timezone: str = "UTC"
    synthetic: SyntheticDataConfig = Field(default_factory=SyntheticDataConfig)


class FeatureConfig(BaseModel):
    lags: list[int] = Field(default_factory=lambda: [1, 2, 5, 10])
    rolling_windows: list[int] = Field(default_factory=lambda: [5, 10, 20, 60])
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    zscore_window: int = 20
    volatility_window: int = 20
    drawdown_window: int = 60
    min_history: int = 60


class LabelConfig(BaseModel):
    target: Literal[
        "next_direction",
        "next_return",
        "multi_horizon_return",
        "edge_after_cost",
        "threshold_action",
    ] = "next_direction"
    horizon: int = 1
    horizons: list[int] = Field(default_factory=lambda: [1, 5, 10])
    cost_buffer_bps: float = 15.0
    action_threshold: float = 0.001


class ModelConfig(BaseModel):
    name: Literal["logistic_regression", "random_forest", "gradient_boosting", "ensemble"] = (
        "logistic_regression"
    )
    task: Literal["classification", "regression"] = "classification"
    calibrate: bool = True
    hyperparameter_search: bool = False
    random_state: int = 7
    max_iter: int = 1000
    cv_splits: int = 3
    model_dir: str = "artifacts/models"


class StrategyConfig(BaseModel):
    long_only: bool = True
    probability_buy_threshold: float = 0.55
    probability_sell_threshold: float = 0.45
    expected_return_threshold: float = 0.0005
    cost_buffer_bps: float = 10.0
    confidence_threshold: float = 0.05
    no_trade_band: float = 0.02
    cooldown_bars_after_loss_streak: int = 3
    max_loss_streak: int = 3


class ExecutionConfig(BaseModel):
    commission_per_order: float = 1.0
    fee_bps: float = 2.0
    spread_bps: float = 5.0
    slippage_bps: float = 3.0
    market_impact_bps: float = 1.0
    allow_partial_fills: bool = False


class TaxConfig(BaseModel):
    enabled: bool = True
    mode: Literal["immediate", "accrued"] = "accrued"
    flat_rate: float = 0.25
    short_term_rate: float = 0.30
    long_term_rate: float = 0.18
    long_term_days: int = 365
    net_losses: bool = True


class RiskConfig(BaseModel):
    max_position_fraction: float = 0.25
    max_gross_exposure: float = 1.0
    sizing_mode: Literal["fixed_fractional", "volatility_adjusted"] = "fixed_fractional"
    fixed_fraction: float = 0.15
    target_volatility: float = 0.10
    max_trades_per_day: int = 10
    max_daily_loss_fraction: float = 0.03
    max_drawdown_fraction: float = 0.20
    kill_switch: bool = False
    human_enabled: bool = False


class WalkForwardConfig(BaseModel):
    train_bars: int = 252
    validation_bars: int = 63
    test_bars: int = 63
    step_bars: int = 63
    retrain_frequency_bars: int = 21


class BacktestConfig(BaseModel):
    initial_cash: float = 100_000.0
    benchmark: list[str] = Field(
        default_factory=lambda: ["buy_and_hold", "random", "moving_average_crossover", "momentum"]
    )
    retrain_frequency_bars: int = 21
    min_train_bars: int = 252
    results_dir: str = "artifacts/runs"
    seed: int = 7


class BrokerConfig(BaseModel):
    mode: Literal["simulation", "paper", "live"] = "simulation"
    provider: Literal["mock", "alpaca", "ibkr"] = "mock"
    paper_guard_enabled: bool = True
    live_trading_enabled: bool = False


class PlatformConfig(BaseModel):
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    tax: TaxConfig = Field(default_factory=TaxConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)

    @field_validator("market_data")
    @classmethod
    def validate_market_data(cls, value: MarketDataConfig) -> MarketDataConfig:
        if value.source == "csv" and not value.csv_path:
            raise ValueError("csv_path must be set when source='csv'")
        return value


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
