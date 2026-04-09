from __future__ import annotations

from trading_platform.app import load_platform_config, run_backtest


def main() -> None:
    config = load_platform_config(profile="synthetic")
    run_backtest(config)


if __name__ == "__main__":
    main()
