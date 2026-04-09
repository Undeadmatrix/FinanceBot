from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_platform.broker.broker_models import Order
from trading_platform.data.dataset_builder import DatasetBundle
from trading_platform.env.execution_engine import ExecutionEngine
from trading_platform.env.portfolio import Portfolio
from trading_platform.env.tax_engine import TaxEngine
from trading_platform.models.selector import build_model
from trading_platform.models.trainer import ModelTrainer
from trading_platform.strategies.alpha_model_strategy import AlphaModelStrategy
from trading_platform.strategies.benchmark_strategies import build_benchmark_strategy
from trading_platform.strategies.position_sizer import PositionSizer
from trading_platform.strategies.risk_manager import RiskManager, RiskState
from trading_platform.strategies.signal_policy import ProbabilitySignalPolicy, StrategyContext
from trading_platform.strategies.trade_filters import TradeFilterEngine
from trading_platform.utils.serialization import dump_json
from trading_platform.utils.validation import (
    BacktestConfig,
    BrokerConfig,
    ExecutionConfig,
    ModelConfig,
    RiskConfig,
    StrategyConfig,
    TaxConfig,
)


@dataclass(slots=True)
class BacktestResult:
    snapshots: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame
    metrics: dict[str, Any]
    benchmark_results: dict[str, "BacktestResult"]
    feature_importance: pd.Series | None = None


class BacktestEngine:
    def __init__(
        self,
        backtest_config: BacktestConfig,
        execution_config: ExecutionConfig,
        tax_config: TaxConfig,
        risk_config: RiskConfig,
        broker_config: BrokerConfig,
        strategy_config: StrategyConfig,
        model_config: ModelConfig,
    ) -> None:
        self.backtest_config = backtest_config
        self.execution_config = execution_config
        self.tax_config = tax_config
        self.risk_config = risk_config
        self.broker_config = broker_config
        self.strategy_config = strategy_config
        self.model_config = model_config

    def run(
        self,
        bundle: DatasetBundle,
        output_dir: str | Path | None = None,
        include_benchmarks: bool = True,
    ) -> BacktestResult:
        model_result = self._run_model_strategy(bundle)
        benchmark_results: dict[str, BacktestResult] = {}
        benchmark_reference = None

        if include_benchmarks:
            for benchmark_name in self.backtest_config.benchmark:
                benchmark_result = self.run_benchmark_strategy(bundle, benchmark_name)
                benchmark_results[benchmark_name] = benchmark_result
                if benchmark_name == "buy_and_hold":
                    benchmark_reference = benchmark_result

        if benchmark_reference is not None:
            from trading_platform.backtest.metrics import compute_performance_metrics

            model_result.metrics = compute_performance_metrics(
                model_result.snapshots,
                model_result.trades,
                benchmark_snapshots=benchmark_reference.snapshots,
            )
        model_result.benchmark_results = benchmark_results

        if output_dir is not None:
            self._persist(output_dir, model_result)
        return model_result

    def _run_model_strategy(self, bundle: DatasetBundle) -> BacktestResult:
        dataset = bundle.dataset.sort_values(["timestamp", "instrument"]).reset_index(drop=True)
        feature_columns = bundle.feature_columns
        dates = sorted(dataset["timestamp"].unique())
        portfolio = Portfolio(self.backtest_config.initial_cash, TaxEngine(self.tax_config))
        execution_engine = ExecutionEngine(self.execution_config)
        trainer = ModelTrainer(self.model_config)
        strategy = AlphaModelStrategy(
            policy=ProbabilitySignalPolicy(self.strategy_config),
            filters=TradeFilterEngine(self.strategy_config),
            sizer=PositionSizer(self.risk_config),
            risk_manager=RiskManager(self.risk_config, self.broker_config),
        )

        signal_rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []
        pending_targets: dict[pd.Timestamp, list[dict[str, Any]]] = defaultdict(list)
        loss_streaks: dict[str, int] = defaultdict(int)
        model = None
        last_train_index = -1
        daily_equity_reference = self.backtest_config.initial_cash
        feature_importance = None

        for idx, date in enumerate(dates):
            bars_today = dataset[dataset["timestamp"] == date].copy()
            trade_count_today = 0
            previous_snapshot = portfolio.equity_history[-1] if portfolio.equity_history else None

            for pending in pending_targets.pop(date, []):
                instrument = pending["instrument"]
                target_fraction = float(pending["target_fraction"])
                bar = bars_today[bars_today["instrument"] == instrument]
                if bar.empty:
                    continue
                bar_row = bar.iloc[0]
                reference_equity = previous_snapshot.equity if previous_snapshot is not None else self.backtest_config.initial_cash
                current_position = portfolio.position_snapshot(instrument)
                current_value = current_position.quantity * float(bar_row["open"])
                desired_value = reference_equity * target_fraction
                delta_value = desired_value - current_value
                if abs(delta_value) < 1e-8:
                    continue
                side = "buy" if delta_value > 0 else "sell"
                quantity = self._calculate_order_quantity(
                    side=side,
                    bar_row=bar_row,
                    delta_value=delta_value,
                    available_cash=portfolio.cash,
                    current_quantity=current_position.quantity,
                )
                if quantity <= 1e-8:
                    continue
                order = Order(
                    timestamp=pd.Timestamp(date),
                    instrument=instrument,
                    side=side,
                    quantity=quantity,
                    metadata={
                        "generated_at": pending["generated_at"],
                        "signal_reason": pending["reason"],
                    },
                )
                fill = execution_engine.execute_market_order(order, bar_row)
                trade = portfolio.process_fill(fill)
                trade_rows.append(asdict(trade))
                trade_count_today += 1
                if trade.realized_pnl < 0:
                    loss_streaks[instrument] += 1
                elif trade.realized_pnl > 0:
                    loss_streaks[instrument] = 0

            close_prices = {row.instrument: float(row.close) for row in bars_today.itertuples(index=False)}
            snapshot = portfolio.mark_to_market(pd.Timestamp(date), close_prices)
            daily_pnl_fraction = snapshot.equity / daily_equity_reference - 1.0 if daily_equity_reference else 0.0
            daily_equity_reference = snapshot.equity

            if idx == len(dates) - 1:
                continue

            train_dates = dates[:idx]
            if len(train_dates) < self.backtest_config.min_train_bars:
                continue

            if model is None or idx - last_train_index >= self.backtest_config.retrain_frequency_bars:
                train_set = dataset[dataset["timestamp"].isin(train_dates)]
                model = trainer.train(
                    build_model(self.model_config),
                    train_set[feature_columns],
                    train_set[bundle.target_column],
                ).model
                last_train_index = idx
                feature_importance = model.feature_importance(feature_columns)

            next_date = dates[idx + 1]
            risk_state = RiskState(
                gross_exposure=snapshot.gross_exposure,
                drawdown=snapshot.drawdown,
                daily_pnl_fraction=daily_pnl_fraction,
                trades_today=trade_count_today,
                human_enabled=self.risk_config.human_enabled,
            )

            for row in bars_today.to_dict(orient="records"):
                current_position = portfolio.position_snapshot(row["instrument"])
                current_fraction = current_position.market_value / snapshot.equity if snapshot.equity else 0.0
                features = pd.DataFrame([{column: row[column] for column in feature_columns}])
                if self.model_config.task == "classification":
                    predicted_probability = float(np.asarray(model.predict_proba(features)).ravel()[0])
                    expected_return = self._estimate_expected_return(row, predicted_probability)
                else:
                    prediction = float(np.asarray(model.predict(features)).ravel()[0])
                    expected_return = prediction
                    predicted_probability = float(1.0 / (1.0 + np.exp(-prediction)))

                estimated_cost_rate = (
                    float(row.get("spread_bps", self.execution_config.spread_bps))
                    + self.execution_config.slippage_bps
                    + self.execution_config.fee_bps
                    + self.execution_config.market_impact_bps
                ) / 10_000.0
                context = StrategyContext(
                    timestamp=pd.Timestamp(date),
                    instrument=row["instrument"],
                    predicted_probability=predicted_probability,
                    expected_return=expected_return,
                    estimated_cost_rate=estimated_cost_rate,
                    realized_volatility=float(row.get("rolling_vol_20", 0.01) or 0.01),
                    current_fraction=current_fraction,
                    current_drawdown=snapshot.drawdown,
                    consecutive_losses=loss_streaks[row["instrument"]],
                    trades_today=trade_count_today,
                    human_enabled=self.risk_config.human_enabled,
                )
                decision = strategy.generate_decision(context, risk_state)
                signal_rows.append(
                    {
                        "timestamp": pd.Timestamp(date),
                        "execution_timestamp": pd.Timestamp(next_date),
                        "instrument": row["instrument"],
                        "predicted_probability": predicted_probability,
                        "expected_return": expected_return,
                        "target_fraction": decision.target_fraction,
                        "action": decision.action,
                        "reason": decision.reason,
                        "blocked": decision.blocked,
                    }
                )
                pending_targets[next_date].append(
                    {
                        "instrument": row["instrument"],
                        "target_fraction": decision.target_fraction,
                        "generated_at": pd.Timestamp(date),
                        "reason": decision.reason,
                    }
                )

        snapshots_df = pd.DataFrame(asdict(snapshot) for snapshot in portfolio.equity_history)
        trades_df = pd.DataFrame(trade_rows)
        signals_df = pd.DataFrame(signal_rows)

        from trading_platform.backtest.metrics import compute_performance_metrics

        metrics = compute_performance_metrics(snapshots_df, trades_df)
        return BacktestResult(
            snapshots=snapshots_df,
            trades=trades_df,
            signals=signals_df,
            metrics=metrics,
            benchmark_results={},
            feature_importance=feature_importance,
        )

    def run_benchmark_strategy(self, bundle: DatasetBundle, benchmark_name: str) -> BacktestResult:
        dataset = bundle.dataset.sort_values(["timestamp", "instrument"]).reset_index(drop=True)
        dates = sorted(dataset["timestamp"].unique())
        strategy = build_benchmark_strategy(benchmark_name, seed=self.backtest_config.seed)
        portfolio = Portfolio(self.backtest_config.initial_cash, TaxEngine(self.tax_config))
        execution_engine = ExecutionEngine(self.execution_config)
        pending_targets: dict[pd.Timestamp, list[dict[str, Any]]] = defaultdict(list)
        signal_rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []

        for idx, date in enumerate(dates):
            bars_today = dataset[dataset["timestamp"] == date].copy()
            previous_snapshot = portfolio.equity_history[-1] if portfolio.equity_history else None

            for pending in pending_targets.pop(date, []):
                instrument = pending["instrument"]
                target_fraction = float(pending["target_fraction"])
                bar = bars_today[bars_today["instrument"] == instrument]
                if bar.empty:
                    continue
                bar_row = bar.iloc[0]
                reference_equity = previous_snapshot.equity if previous_snapshot is not None else self.backtest_config.initial_cash
                current_position = portfolio.position_snapshot(instrument)
                current_value = current_position.quantity * float(bar_row["open"])
                desired_value = reference_equity * target_fraction
                delta_value = desired_value - current_value
                if abs(delta_value) < 1e-8:
                    continue
                side = "buy" if delta_value > 0 else "sell"
                quantity = self._calculate_order_quantity(
                    side=side,
                    bar_row=bar_row,
                    delta_value=delta_value,
                    available_cash=portfolio.cash,
                    current_quantity=current_position.quantity,
                )
                if quantity <= 1e-8:
                    continue
                fill = execution_engine.execute_market_order(
                    Order(timestamp=pd.Timestamp(date), instrument=instrument, side=side, quantity=quantity),
                    bar_row,
                )
                trade_rows.append(asdict(portfolio.process_fill(fill)))

            close_prices = {row.instrument: float(row.close) for row in bars_today.itertuples(index=False)}
            portfolio.mark_to_market(pd.Timestamp(date), close_prices)
            if idx == len(dates) - 1:
                continue
            next_date = dates[idx + 1]
            for row in bars_today.to_dict(orient="records"):
                if benchmark_name == "buy_and_hold":
                    current_position = portfolio.position_snapshot(row["instrument"])
                    if abs(current_position.quantity) > 1e-9:
                        continue
                    decision = strategy.generate_decision(row["instrument"])
                elif benchmark_name == "random":
                    decision = strategy.generate_decision(row["instrument"])
                else:
                    decision = strategy.generate_decision(row, row["instrument"])
                signal_rows.append(
                    {
                        "timestamp": pd.Timestamp(date),
                        "execution_timestamp": pd.Timestamp(next_date),
                        "instrument": row["instrument"],
                        "predicted_probability": decision.predicted_probability,
                        "expected_return": decision.expected_return,
                        "target_fraction": decision.target_fraction,
                        "action": decision.action,
                        "reason": decision.reason,
                        "blocked": decision.blocked,
                    }
                )
                pending_targets[next_date].append(
                    {
                        "instrument": row["instrument"],
                        "target_fraction": decision.target_fraction,
                    }
                )

        snapshots_df = pd.DataFrame(asdict(snapshot) for snapshot in portfolio.equity_history)
        trades_df = pd.DataFrame(trade_rows)
        signals_df = pd.DataFrame(signal_rows)

        from trading_platform.backtest.metrics import compute_performance_metrics

        metrics = compute_performance_metrics(snapshots_df, trades_df)
        return BacktestResult(
            snapshots=snapshots_df,
            trades=trades_df,
            signals=signals_df,
            metrics=metrics,
            benchmark_results={},
            feature_importance=None,
        )

    def _estimate_expected_return(self, row: dict[str, Any], probability: float) -> float:
        realized_vol = max(float(row.get("rolling_vol_20", 0.01) or 0.01), 0.0025)
        trend_component = float(row.get("rolling_return_5", 0.0) or 0.0) * 0.25
        return (probability - 0.5) * 2.0 * realized_vol + trend_component

    def _calculate_order_quantity(
        self,
        side: str,
        bar_row: pd.Series,
        delta_value: float,
        available_cash: float,
        current_quantity: float,
    ) -> float:
        open_price = float(bar_row["open"])
        spread_bps = float(bar_row.get("spread_bps", self.execution_config.spread_bps))
        side_sign = 1.0 if side == "buy" else -1.0
        estimated_fill = open_price + open_price * (
            (spread_bps / 20_000.0) + (self.execution_config.slippage_bps / 10_000.0) + (self.execution_config.market_impact_bps / 10_000.0)
        ) * side_sign
        estimated_fill = max(estimated_fill, 1e-8)

        if side == "buy":
            gross_budget = min(abs(delta_value), max(0.0, available_cash - self.execution_config.commission_per_order))
            if gross_budget <= 0.0:
                return 0.0
            return gross_budget / (estimated_fill * (1.0 + self.execution_config.fee_bps / 10_000.0))

        return min(current_quantity, abs(delta_value) / estimated_fill)

    def _persist(self, output_dir: str | Path, result: BacktestResult) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result.snapshots.to_csv(output_path / "snapshots.csv", index=False)
        result.trades.to_csv(output_path / "trades.csv", index=False)
        result.signals.to_csv(output_path / "signals.csv", index=False)
        dump_json(output_path / "metrics.json", result.metrics)
        if result.feature_importance is not None:
            result.feature_importance.to_csv(output_path / "feature_importance.csv", header=["importance"])
