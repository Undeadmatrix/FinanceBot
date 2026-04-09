from __future__ import annotations

from trading_platform.app import load_platform_config, run_walk_forward


def main() -> None:
    config = load_platform_config(profile="backtest")
    run_walk_forward(config)


if __name__ == "__main__":
    main()
