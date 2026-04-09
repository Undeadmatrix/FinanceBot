from __future__ import annotations

from trading_platform.utils.validation import BrokerConfig, RiskConfig


def assert_live_trading_allowed(broker_config: BrokerConfig, risk_config: RiskConfig) -> None:
    if broker_config.mode == "live":
        if not broker_config.live_trading_enabled:
            raise PermissionError("Live trading is disabled by configuration")
        if not risk_config.human_enabled:
            raise PermissionError("Human enablement flag is required for live trading")


def assert_paper_trading_allowed(broker_config: BrokerConfig, risk_config: RiskConfig) -> None:
    if broker_config.mode == "paper" and broker_config.paper_guard_enabled and not risk_config.human_enabled:
        raise PermissionError("Paper trading requires the explicit human-enabled flag")
