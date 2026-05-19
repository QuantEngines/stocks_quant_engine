"""Schemas for sector intelligence and rotation reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SectorIntelligenceReport:
    sector: str
    sector_score: float
    momentum_score: float
    fundamentals_score: float
    valuation_score: float
    risk_score: float
    event_macro_score: float
    stance: str
    best_long_term_stocks: list[str] = field(default_factory=list)
    best_swing_candidates: list[str] = field(default_factory=list)
    avoid_watchlist_stocks: list[str] = field(default_factory=list)
    thesis: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    monitorables: list[str] = field(default_factory=list)
    coverage: dict[str, float | int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
