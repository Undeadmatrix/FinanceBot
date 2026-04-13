# This is just me experimenting with the capabilities of v*be coding

`trading_platform` is a research-oriented quantitative trading framework built to behave more like a small internal quant team’s platform than a hobby bot. Version 1 starts in a fully simulated setting, supports historical CSV backtesting, includes walk-forward validation, exposes a guarded paper-trading path, and keeps live execution behind explicit safety gates.

The design goal is research integrity first:

- No fake profitability claims.
- No random shuffling of time series.
- No same-bar impossible fills.
- No assumption that predictive accuracy implies trading profitability.
- No evaluation before fees, spread, slippage, and tax drag.

## Architecture

The package is organized as a layered research and execution stack:

- `trading_platform/config`: YAML profiles for default, synthetic, backtest, and paper modes.
- `trading_platform/data`: synthetic market generation, CSV loading, resampling, feature engineering, labeling, and dataset assembly.
- `trading_platform/models`: typed model abstractions, logistic regression, tree models, calibration, training, and model selection.
- `trading_platform/strategies`: probability-to-signal policy, trade filters, sizing, risk vetoes, and benchmark strategies.
- `trading_platform/env`: execution cost modeling, market simulation, tax handling, accounting, and portfolio state.
- `trading_platform/backtest`: sequential backtest engine, benchmarks, walk-forward validation, metrics, attribution, and tear-sheet generation.
- `trading_platform/broker`: broker interfaces, guarded paper broker, and live adapter skeletons for Alpaca and IBKR.
- `trading_platform/monitoring`: run metadata, config snapshots, logging, and simple alert hooks.
- `trading_platform/tests`: leakage, execution, portfolio, tax, backtest, walk-forward, and integration tests.

## Research Workflow

The intended workflow is:

1. Generate synthetic multi-regime data.
2. Build leakage-safe features and forward labels.
3. Train a probabilistic model on past data only.
4. Convert model outputs into cost-aware target exposure through the strategy layer.
5. Simulate next-bar execution with fees, spread, slippage, and tax treatment.
6. Compare against benchmarks.
7. Run walk-forward validation to see when the model works, and when it breaks.
8. Move to historical CSV backtests.
9. Move to guarded paper simulation only after the research path is reproducible.

## Realism Controls

The system enforces realism in several ways:

- Features use only current and past bars.
- Labels are forward-looking and unresolved tail rows are dropped.
- The backtest loop is sequential and bar-by-bar.
- Signals are generated on bar `t` and only execute on bar `t+1` open.
- Execution prices are shifted away from the raw open by spread, slippage, fees, and market-impact placeholders.
- Taxes are modeled by default with immediate or accrued handling.
- Benchmarks run by default so model behavior is not judged in isolation.
- Synthetic runs are seed-controlled and config snapshots are persisted for reproducibility.

## Version 1 Scope

Implemented in V1:

- Multi-instrument synthetic OHLCV generator with regime shifts, volatility clustering, jumps, liquidity/spread changes, and outlier events.
- Historical CSV loader with schema validation and timezone-safe timestamps.
- Leakage-safe feature pipeline.
- Flexible label builder for classification/regression-style targets.
- Logistic regression baseline plus tree-model wrappers and ensemble placeholder.
- Calibrated probability path for the main model workflow.
- Cost-aware alpha strategy policy with no-trade region, thresholds, filters, sizing, and risk vetoes.
- Execution engine with market-order simulation and default friction.
- Portfolio, accounting, and tax engine.
- Sequential backtesting with benchmark comparison.
- Walk-forward validation.
- Tear-sheet plots for equity, drawdown, and rolling returns.
- Guarded paper-trading simulation path.

## Quick Start

Create a Python 3.11+ environment, install dependencies, and run the synthetic demo:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py run-backtest --profile synthetic
```

Useful commands:

```bash
python main.py generate-synthetic --profile synthetic --output artifacts/synthetic.csv
python main.py build-dataset --profile synthetic --output artifacts/dataset.csv
python main.py train-model --profile backtest --output-dir artifacts/model_training
python main.py run-backtest --profile backtest
python main.py run-walk-forward --profile backtest
python main.py run-paper --profile paper
pytest trading_platform/tests
```

## Configuration

Configuration is YAML-driven. The default profile covers:

- market data source and synthetic generator settings
- feature windows and indicator settings
- label target selection
- model choice and training behavior
- commissions, fees, spread, slippage, and impact assumptions
- tax treatment
- risk rules
- walk-forward windows
- benchmark set
- broker mode and safety flags

You can pass a custom override file with `--config path/to/file.yaml`.

## Adding Features, Models, and Strategies

To add new research components:

- Features: extend [`trading_platform/data/feature_pipeline.py`](/C:/Users/Chris/OneDrive/Desktop/DEVSHIT/FinanceBot/trading_platform/data/feature_pipeline.py)
- Labels: extend [`trading_platform/data/labeler.py`](/C:/Users/Chris/OneDrive/Desktop/DEVSHIT/FinanceBot/trading_platform/data/labeler.py)
- Models: implement the estimator interface in [`trading_platform/models/base.py`](/C:/Users/Chris/OneDrive/Desktop/DEVSHIT/FinanceBot/trading_platform/models/base.py) and register it in [`trading_platform/models/selector.py`](/C:/Users/Chris/OneDrive/Desktop/DEVSHIT/FinanceBot/trading_platform/models/selector.py)
- Strategy policy: extend [`trading_platform/strategies/signal_policy.py`](/C:/Users/Chris/OneDrive/Desktop/DEVSHIT/FinanceBot/trading_platform/strategies/signal_policy.py)
- Risk rules: extend [`trading_platform/strategies/risk_manager.py`](/C:/Users/Chris/OneDrive/Desktop/DEVSHIT/FinanceBot/trading_platform/strategies/risk_manager.py)

Keep any new component time-safe, reproducible, and benchmarked against simple alternatives.

## Safety Gates

Paper/live separation is deliberate:

- `simulation` is the default mode.
- `paper` mode requires the `human_enabled` flag.
- `live` placement is disabled unless both broker and human safety flags explicitly allow it.
- Alpaca and IBKR adapters are skeletons in V1 and do not place live orders by default.

## Why Backtest Performance Is Not Proof Of Alpha

Backtests can still fail for many reasons even if the code is careful:

- synthetic data can reward the wrong inductive bias
- historical relationships can decay
- real spreads and slippage can be worse than assumed
- taxes and operational constraints can dominate small model edges
- model calibration can drift by regime

This framework is built to surface those failures earlier, not to promise they can be avoided.

## Limitations

- simulated success does not imply real-market profitability
- professional firms have superior data, infrastructure, and execution
- this codebase is a research framework, not a guarantee of edge
