"""Schemas for sector-relative peer comparison reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class PeerComparisonRow:
    symbol: str
    company_name: str
    sector: str
    industry: str
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    earnings_yield: float | None = None
    revenue_growth_yoy: float | None = None
    earnings_growth_yoy: float | None = None
    roe: float | None = None
    roce: float | None = None
    operating_margin: float | None = None
    net_profit_margin: float | None = None
    free_cash_flow_margin: float | None = None
    debt_to_equity: float | None = None
    interest_coverage: float | None = None
    valuation_score: float = 0.0
    quality_score: float = 0.0
    growth_score: float = 0.0
    risk_score: float = 0.0
    composite_score: float = 0.0
    valuation_rank: int | None = None
    quality_rank: int | None = None
    growth_rank: int | None = None
    risk_rank: int | None = None
    composite_rank: int | None = None
    missing_data: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PeerComparisonReport:
    symbol: str
    company_name: str
    sector: str
    industry: str
    as_of: str
    peer_count: int
    target: PeerComparisonRow | None
    peers: list[PeerComparisonRow]
    valuation_leaders: list[str] = field(default_factory=list)
    quality_leaders: list[str] = field(default_factory=list)
    growth_leaders: list[str] = field(default_factory=list)
    risk_leaders: list[str] = field(default_factory=list)
    composite_leaders: list[str] = field(default_factory=list)
    thesis: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["target"] = self.target.to_dict() if self.target is not None else None
        payload["peers"] = [peer.to_dict() for peer in self.peers]
        return payload


@dataclass(frozen=True)
class SectorPeerComparisonReport:
    sector: str
    as_of: str
    peer_count: int
    peers: list[PeerComparisonRow]
    valuation_leaders: list[str] = field(default_factory=list)
    quality_leaders: list[str] = field(default_factory=list)
    growth_leaders: list[str] = field(default_factory=list)
    risk_leaders: list[str] = field(default_factory=list)
    composite_leaders: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["peers"] = [peer.to_dict() for peer in self.peers]
        return payload
