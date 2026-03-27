"""Rich signal schemas for final candidate outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class SignalDrivers:
    top_positive: list[str] = field(default_factory=list)
    top_negative: list[str] = field(default_factory=list)
    missing_features: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RankedSignal:
    symbol: str
    stock_name: str | None
    signal_category: str
    as_of: date
    long_term_score: float
    swing_score: float
    risk_penalty: float
    final_score: float
    rank: int
    conviction: float
    horizon: str
    drivers: SignalDrivers
    rejection_reasons: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    sector: str = ""
    invalidation_notes: list[str] = field(default_factory=list)
    regime: str = "unknown"

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict for storage and reporting."""
        return {
            "symbol": self.symbol,
            "stock_name": self.stock_name,
            "signal_category": self.signal_category,
            "as_of": self.as_of.isoformat(),
            "sector": self.sector,
            "regime": self.regime,
            "long_term_score": round(self.long_term_score, 2),
            "swing_score": round(self.swing_score, 2),
            "risk_penalty": round(self.risk_penalty, 2),
            "final_score": round(self.final_score, 2),
            "conviction": round(self.conviction, 2),
            "horizon": self.horizon,
            "rank": self.rank,
            "top_positive_drivers": list(self.drivers.top_positive),
            "top_negative_drivers": list(self.drivers.top_negative),
            "missing_features": list(self.drivers.missing_features),
            "rejection_reasons": list(self.rejection_reasons),
            "risk_flags": list(self.risk_flags),
            "invalidation_notes": list(self.invalidation_notes),
        }
