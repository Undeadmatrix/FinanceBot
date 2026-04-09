from __future__ import annotations

import logging


class AlertManager:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def risk_breach(self, message: str) -> None:
        self.logger.warning("RISK ALERT: %s", message)

    def trading_halted(self, message: str) -> None:
        self.logger.error("TRADING HALTED: %s", message)
