from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from trading_platform.backtest.engine import BacktestEngine
from trading_platform.backtest.tearsheet import TearSheetGenerator
from trading_platform.backtest.walk_forward import WalkForwardValidator
from trading_platform.broker.paper_broker import PaperBroker
from trading_platform.data.dataset_builder import DatasetBuilder, DatasetBundle
from trading_platform.data.feature_pipeline import FeaturePipeline
from trading_platform.data.labeler import LabelBuilder
from trading_platform.data.market_data_loader import MarketDataLoader
from trading_platform.data.synthetic_generator import generate_synthetic_market
from trading_platform.models.selector import build_model
from trading_platform.models.trainer import ModelTrainer
from trading_platform.monitoring.experiment_logger import ExperimentLogger, configure_logging
from trading_platform.strategies.alpha_model_strategy import AlphaModelStrategy
from trading_platform.strategies.position_sizer import PositionSizer
from trading_platform.strategies.risk_manager import RiskManager, RiskState
from trading_platform.strategies.signal_policy import ProbabilitySignalPolicy, StrategyContext
from trading_platform.strategies.trade_filters import TradeFilterEngine
from trading_platform.utils.reporting import write_backtest_report, write_paper_report, write_walk_forward_report
from trading_platform.utils.serialization import deep_merge, dump_json, load_yaml, model_to_dict
from trading_platform.utils.validation import PlatformConfig


LOGGER = logging.getLogger(__name__)


def load_platform_config(profile: str | None = None, config_path: str | None = None) -> PlatformConfig:
    config_dir = Path(__file__).resolve().parent / "config"
    payload = load_yaml(config_dir / "default.yaml")
    if profile:
        profile_path = config_dir / f"{profile}.yaml"
        if profile_path.exists():
            payload = deep_merge(payload, load_yaml(profile_path))
        else:
            raise FileNotFoundError(f"Unknown profile: {profile}")
    if config_path:
        payload = deep_merge(payload, load_yaml(config_path))
    return PlatformConfig(**payload)


def load_market_data(config: PlatformConfig) -> pd.DataFrame:
    if config.market_data.source == "synthetic":
        return generate_synthetic_market(config.market_data.synthetic)
    loader = MarketDataLoader(timezone=config.market_data.timezone)
    return loader.load_csv(config.market_data.csv_path)


def build_dataset_bundle(config: PlatformConfig, frame: pd.DataFrame | None = None) -> DatasetBundle:
    frame = frame if frame is not None else load_market_data(config)
    feature_pipeline = FeaturePipeline(config.features)
    label_builder = LabelBuilder(config.labels, config.execution)
    builder = DatasetBuilder(feature_pipeline, label_builder)
    return builder.build(frame)


def train_model(bundle: DatasetBundle, config: PlatformConfig, output_dir: str | Path | None = None) -> dict[str, Any]:
    builder = DatasetBuilder(FeaturePipeline(config.features), LabelBuilder(config.labels, config.execution))
    train_set, validation_set, test_set = builder.time_split(bundle.dataset)
    trainer = ModelTrainer(config.model)
    result = trainer.train(
        build_model(config.model),
        train_set[bundle.feature_columns],
        train_set[bundle.target_column],
    )
    model = result.model
    model_path = None
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        model_path = model.save(output / "model.joblib")
        feature_importance = model.feature_importance(bundle.feature_columns)
        feature_importance.to_csv(output / "feature_importance.csv", header=["importance"])
        dump_json(
            output / "train_summary.json",
            {
                "train_rows": len(train_set),
                "validation_rows": len(validation_set),
                "test_rows": len(test_set),
                "model_path": str(model_path),
            },
        )
    return {"model": model, "model_path": model_path, "train_rows": len(train_set), "validation_rows": len(validation_set), "test_rows": len(test_set)}


def run_backtest(config: PlatformConfig, output_dir: str | Path | None = None, include_benchmarks: bool = True):
    bundle = build_dataset_bundle(config)
    experiment = ExperimentLogger(config.backtest.results_dir)
    run = experiment.start_run(config, prefix="backtest")
    target_dir = Path(output_dir) if output_dir is not None else run.run_dir
    engine = BacktestEngine(
        backtest_config=config.backtest,
        execution_config=config.execution,
        tax_config=config.tax,
        risk_config=config.risk,
        broker_config=config.broker,
        strategy_config=config.strategy,
        model_config=config.model,
    )
    result = engine.run(bundle, output_dir=target_dir, include_benchmarks=include_benchmarks)
    benchmark = result.benchmark_results.get("buy_and_hold") if result.benchmark_results else None
    TearSheetGenerator(target_dir).generate(
        result.snapshots,
        result.trades,
        benchmark_snapshots=benchmark.snapshots if benchmark is not None else None,
        prefix="backtest",
    )
    if result.feature_importance is not None:
        result.feature_importance.to_csv(Path(target_dir) / "feature_importance.csv", header=["importance"])
    write_backtest_report(
        target_dir,
        metrics=result.metrics,
        trades=result.trades,
        snapshots=result.snapshots,
        benchmark_results=result.benchmark_results,
        feature_importance=result.feature_importance,
    )
    LOGGER.info("Backtest completed in %s", target_dir)
    return result, target_dir


def run_walk_forward(config: PlatformConfig, output_dir: str | Path | None = None) -> tuple[pd.DataFrame, Path]:
    bundle = build_dataset_bundle(config)
    experiment = ExperimentLogger(config.backtest.results_dir)
    run = experiment.start_run(config, prefix="walk_forward")
    target_dir = Path(output_dir) if output_dir is not None else run.run_dir
    summary = WalkForwardValidator(config).run(bundle, output_dir=target_dir)
    write_walk_forward_report(target_dir, summary)
    LOGGER.info("Walk-forward summary saved to %s", target_dir)
    return summary, target_dir


def run_paper_simulation(config: PlatformConfig, output_dir: str | Path | None = None) -> dict[str, Any]:
    bundle = build_dataset_bundle(config)
    dataset = bundle.dataset.sort_values(["timestamp", "instrument"]).reset_index(drop=True)
    unique_dates = sorted(dataset["timestamp"].unique())
    if len(unique_dates) < config.backtest.min_train_bars + 2:
        raise ValueError("Not enough history to run the guarded paper simulation")

    execution_date = unique_dates[-1]
    signal_date = unique_dates[-2]
    train_dates = unique_dates[:-2]
    train_set = dataset[dataset["timestamp"].isin(train_dates)]
    signal_rows = dataset[dataset["timestamp"] == signal_date]

    trainer = ModelTrainer(config.model)
    model = trainer.train(
        build_model(config.model),
        train_set[bundle.feature_columns],
        train_set[bundle.target_column],
    ).model

    strategy = AlphaModelStrategy(
        policy=ProbabilitySignalPolicy(config.strategy),
        filters=TradeFilterEngine(config.strategy),
        sizer=PositionSizer(config.risk),
        risk_manager=RiskManager(config.risk, config.broker),
    )
    broker = PaperBroker(config.broker, config.risk)

    results = []
    for row in signal_rows.to_dict(orient="records"):
        features = pd.DataFrame([{column: row[column] for column in bundle.feature_columns}])
        probability = float(model.predict_proba(features)[0])
        expected_return = (probability - 0.5) * 2.0 * max(float(row.get("rolling_vol_20", 0.01) or 0.01), 0.0025)
        context = StrategyContext(
            timestamp=pd.Timestamp(signal_date),
            instrument=row["instrument"],
            predicted_probability=probability,
            expected_return=expected_return,
            estimated_cost_rate=(float(row.get("spread_bps", config.execution.spread_bps)) + config.execution.fee_bps) / 10_000.0,
            realized_volatility=float(row.get("rolling_vol_20", 0.01) or 0.01),
            current_fraction=0.0,
            current_drawdown=0.0,
            consecutive_losses=0,
            trades_today=0,
            human_enabled=config.risk.human_enabled,
        )
        decision = strategy.generate_decision(
            context,
            RiskState(
                gross_exposure=0.0,
                drawdown=0.0,
                daily_pnl_fraction=0.0,
                trades_today=0,
                human_enabled=config.risk.human_enabled,
            ),
        )
        quantity = max(0.0, decision.target_fraction) * 100.0
        status = None
        if quantity > 0.0:
            from trading_platform.broker.broker_models import Order

            status = broker.submit_order(
                Order(timestamp=pd.Timestamp(execution_date), instrument=row["instrument"], side="buy", quantity=quantity)
            )
        results.append(
            {
                "instrument": row["instrument"],
                "signal_date": str(signal_date),
                "execution_date": str(execution_date),
                "predicted_probability": probability,
                "expected_return": expected_return,
                "target_fraction": decision.target_fraction,
                "decision_reason": decision.reason,
                "order_status": status.status if status is not None else "no_order",
            }
        )

    target_dir = Path(output_dir or Path(config.backtest.results_dir) / "paper_sim")
    target_dir.mkdir(parents=True, exist_ok=True)
    dump_json(target_dir / "paper_simulation.json", results)
    write_paper_report(target_dir, results)
    return {"results": results, "output_dir": str(target_dir)}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Institutional-style quantitative trading research platform")
    parser.add_argument("--profile", default=None, help="Optional built-in profile: synthetic, backtest, paper")
    parser.add_argument("--config", default=None, help="Optional path to a YAML config override")

    subparsers = parser.add_subparsers(dest="command", required=True)

    synth = subparsers.add_parser("generate-synthetic", help="Generate synthetic OHLCV data")
    synth.add_argument("--profile", default=None, help="Optional built-in profile: synthetic, backtest, paper")
    synth.add_argument("--config", default=None, help="Optional path to a YAML config override")
    synth.add_argument("--output", default="artifacts/synthetic_data.csv")

    build = subparsers.add_parser("build-dataset", help="Build feature/label dataset")
    build.add_argument("--profile", default=None, help="Optional built-in profile: synthetic, backtest, paper")
    build.add_argument("--config", default=None, help="Optional path to a YAML config override")
    build.add_argument("--output", default="artifacts/dataset.csv")

    train = subparsers.add_parser("train-model", help="Train the configured model")
    train.add_argument("--profile", default=None, help="Optional built-in profile: synthetic, backtest, paper")
    train.add_argument("--config", default=None, help="Optional path to a YAML config override")
    train.add_argument("--output-dir", default="artifacts/model_training")

    backtest = subparsers.add_parser("run-backtest", help="Run sequential backtest")
    backtest.add_argument("--profile", default=None, help="Optional built-in profile: synthetic, backtest, paper")
    backtest.add_argument("--config", default=None, help="Optional path to a YAML config override")
    backtest.add_argument("--output-dir", default=None)

    walk = subparsers.add_parser("run-walk-forward", help="Run walk-forward validation")
    walk.add_argument("--profile", default=None, help="Optional built-in profile: synthetic, backtest, paper")
    walk.add_argument("--config", default=None, help="Optional path to a YAML config override")
    walk.add_argument("--output-dir", default=None)

    paper = subparsers.add_parser("run-paper", help="Run guarded paper-trading simulation")
    paper.add_argument("--profile", default=None, help="Optional built-in profile: synthetic, backtest, paper")
    paper.add_argument("--config", default=None, help="Optional path to a YAML config override")
    paper.add_argument("--output-dir", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = create_parser()
    args = parser.parse_args(argv)
    config = load_platform_config(profile=args.profile, config_path=args.config)

    if args.command == "generate-synthetic":
        data = load_market_data(config)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output, index=False)
        print(f"Synthetic data written to {output}")
        return 0

    if args.command == "build-dataset":
        bundle = build_dataset_bundle(config)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        bundle.dataset.to_csv(output, index=False)
        print(f"Dataset written to {output}")
        return 0

    if args.command == "train-model":
        bundle = build_dataset_bundle(config)
        result = train_model(bundle, config, output_dir=args.output_dir)
        print(f"Model trained. Output: {result['model_path'] or args.output_dir}")
        return 0

    if args.command == "run-backtest":
        result, target_dir = run_backtest(config, output_dir=args.output_dir)
        print(f"Backtest complete. Results in {target_dir}")
        print(f"Text report: {Path(target_dir) / 'backtest_report.txt'}")
        print(result.metrics)
        return 0

    if args.command == "run-walk-forward":
        summary, target_dir = run_walk_forward(config, output_dir=args.output_dir)
        print(f"Walk-forward complete. Results in {target_dir}")
        print(f"Text report: {Path(target_dir) / 'walk_forward_report.txt'}")
        print(summary.tail())
        return 0

    if args.command == "run-paper":
        paper_result = run_paper_simulation(config, output_dir=args.output_dir)
        print(f"Paper simulation complete. Results in {paper_result['output_dir']}")
        print(f"Text report: {Path(paper_result['output_dir']) / 'paper_report.txt'}")
        return 0

    parser.error("Unknown command")
    return 1
