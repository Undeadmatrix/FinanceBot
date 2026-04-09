from __future__ import annotations

import logging
from dataclasses import dataclass, field


@dataclass(slots=True)
class PerformanceMonitor:
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("PerformanceMonitor"))

    def log_state(self, message: str, **kwargs: object) -> None:
        if kwargs:
            message = f"{message} | " + ", ".join(f"{key}={value}" for key, value in kwargs.items())
        self.logger.info(message)
