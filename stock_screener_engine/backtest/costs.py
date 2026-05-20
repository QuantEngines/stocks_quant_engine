"""Configurable Indian equity transaction-cost model."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class IndianEquityCostModel:
    """Approximate cash-equity round-trip costs.

    Defaults are deliberately configurable approximations, not a broker tariff
    guarantee. Production backtests should replace them with the exact broker,
    segment, and date-specific schedule.
    """

    brokerage_bps_per_side: float = 0.0
    stt_buy_bps: float = 10.0
    stt_sell_bps: float = 10.0
    exchange_txn_bps_per_side: float = 0.35
    sebi_bps_per_side: float = 0.01
    stamp_buy_bps: float = 1.5
    gst_rate: float = 0.18
    slippage_bps_per_side: float = 5.0
    explicit_round_trip_bps: float | None = None

    def round_trip_bps(self) -> float:
        if self.explicit_round_trip_bps is not None:
            return max(0.0, float(self.explicit_round_trip_bps))
        taxable_per_side = self.brokerage_bps_per_side + self.exchange_txn_bps_per_side + self.sebi_bps_per_side
        gst_bps = 2.0 * taxable_per_side * self.gst_rate
        return max(
            0.0,
            2.0 * self.brokerage_bps_per_side
            + self.stt_buy_bps
            + self.stt_sell_bps
            + 2.0 * self.exchange_txn_bps_per_side
            + 2.0 * self.sebi_bps_per_side
            + self.stamp_buy_bps
            + gst_bps
            + 2.0 * self.slippage_bps_per_side,
        )

    def round_trip_fraction(self) -> float:
        return self.round_trip_bps() / 10_000.0

    def net_return(self, gross_return: float) -> float:
        return float(gross_return) - self.round_trip_fraction()

    def to_dict(self) -> dict[str, float | None]:
        payload = asdict(self)
        payload["round_trip_bps"] = self.round_trip_bps()
        payload["round_trip_fraction"] = self.round_trip_fraction()
        return payload
