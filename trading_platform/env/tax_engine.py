from __future__ import annotations

from dataclasses import dataclass

from trading_platform.utils.validation import TaxConfig


@dataclass(slots=True)
class TaxEvent:
    realized_pnl: float
    holding_days: int
    tax_delta: float
    liability_delta: float
    applied_rate: float


class TaxEngine:
    def __init__(self, config: TaxConfig) -> None:
        self.config = config
        self.short_bucket = 0.0
        self.long_bucket = 0.0
        self.tax_liability = 0.0
        self.taxes_paid = 0.0

    def realize(self, pnl: float, holding_days: int) -> TaxEvent:
        if not self.config.enabled or pnl == 0.0:
            return TaxEvent(realized_pnl=pnl, holding_days=holding_days, tax_delta=0.0, liability_delta=0.0, applied_rate=0.0)

        rate = self.config.long_term_rate if holding_days >= self.config.long_term_days else self.config.short_term_rate
        previous_liability = self.tax_liability

        if holding_days >= self.config.long_term_days:
            self.long_bucket += pnl
            if not self.config.net_losses:
                self.long_bucket = max(self.long_bucket, 0.0)
        else:
            self.short_bucket += pnl
            if not self.config.net_losses:
                self.short_bucket = max(self.short_bucket, 0.0)

        self.tax_liability = max(self.short_bucket, 0.0) * self.config.short_term_rate + max(self.long_bucket, 0.0) * self.config.long_term_rate
        liability_delta = self.tax_liability - previous_liability
        tax_delta = liability_delta if self.config.mode == "immediate" else 0.0
        self.taxes_paid += tax_delta
        return TaxEvent(
            realized_pnl=pnl,
            holding_days=holding_days,
            tax_delta=tax_delta,
            liability_delta=liability_delta,
            applied_rate=rate,
        )
