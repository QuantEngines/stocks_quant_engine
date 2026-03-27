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
