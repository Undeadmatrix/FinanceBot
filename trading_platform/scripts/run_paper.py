from __future__ import annotations

from trading_platform.app import load_platform_config, run_paper_simulation


def main() -> None:
    config = load_platform_config(profile="paper")
    run_paper_simulation(config)


if __name__ == "__main__":
    main()
