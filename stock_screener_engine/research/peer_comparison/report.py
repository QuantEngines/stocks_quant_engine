"""Sector-relative peer comparison engine."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date
from typing import Any, Mapping, Sequence, cast

from stock_screener_engine.core.entities import FundamentalsSnapshot
from stock_screener_engine.data_sources.financials.sqlite_financials_provider import SQLiteFinancialsProvider
from stock_screener_engine.data_sources.schemas import SecurityMasterRecord
from stock_screener_engine.research.peer_comparison.schemas import (
    PeerComparisonReport,
    PeerComparisonRow,
    SectorPeerComparisonReport,
)
from stock_screener_engine.storage.market_data_store import MarketDataStore


class PeerComparisonBuilder:
    """Build point-in-time sector peer ranks from canonical fundamentals."""

    def __init__(
        self,
        store: MarketDataStore,
        financials: SQLiteFinancialsProvider,
        venue: str = "NSE",
    ) -> None:
        self.store = store
        self.financials = financials
        self.venue = venue.strip().upper() or "NSE"

    def build(self, symbol: str, as_of: date) -> PeerComparisonReport:
        normalized = symbol.strip().upper()
        target_record = self._security(normalized)
        if target_record is None:
            return PeerComparisonReport(
                symbol=normalized,
                company_name=normalized,
                sector="Unknown",
                industry="Unknown",
                as_of=as_of.isoformat(),
                peer_count=0,
                target=None,
                peers=[],
                warnings=[f"{normalized} not found in canonical security master."],
            )

        sector_report = self.build_sector(target_record.sector, as_of=as_of)
        target = next((row for row in sector_report.peers if row.symbol == normalized), None)
        warnings = list(sector_report.warnings)
        if target is None:
            warnings.append(f"{normalized} has no point-in-time fundamentals available for peer ranking.")

        return PeerComparisonReport(
            symbol=normalized,
            company_name=target_record.company_name or normalized,
            sector=target_record.sector,
            industry=target_record.industry,
            as_of=as_of.isoformat(),
            peer_count=sector_report.peer_count,
            target=target,
            peers=sector_report.peers,
            valuation_leaders=sector_report.valuation_leaders,
            quality_leaders=sector_report.quality_leaders,
            growth_leaders=sector_report.growth_leaders,
            risk_leaders=sector_report.risk_leaders,
            composite_leaders=sector_report.composite_leaders,
            thesis=_target_thesis(target),
            warnings=warnings,
        )

    def build_sector(self, sector: str, as_of: date) -> SectorPeerComparisonReport:
        sector_name = sector.strip() or "Unknown"
        securities = self.store.list_active_securities(sectors=[sector_name], exchange=self.venue)
        symbols = [security.symbol for security in securities]
        if not securities:
            return SectorPeerComparisonReport(
                sector=sector_name,
                as_of=as_of.isoformat(),
                peer_count=0,
                peers=[],
                warnings=[f"No active securities found for sector {sector_name}."],
            )

        fundamentals = self.financials.get_fundamentals_as_of(symbols, as_of=as_of)
        rows = [
            _base_row(security, fundamentals.get(security.symbol))
            for security in securities
            if security.symbol in fundamentals
        ]
        warnings: list[str] = []
        missing_symbols = sorted(set(symbols) - set(fundamentals))
        if missing_symbols:
            warnings.append(
                "Missing point-in-time fundamentals for: "
                + ", ".join(missing_symbols[:10])
            )
        if not rows:
            return SectorPeerComparisonReport(
                sector=sector_name,
                as_of=as_of.isoformat(),
                peer_count=0,
                peers=[],
                warnings=warnings or [f"No peer fundamentals available for sector {sector_name}."],
            )

        scored = _score_rows(rows)
        ranked = sorted(scored, key=lambda row: row.composite_score, reverse=True)
        return SectorPeerComparisonReport(
            sector=sector_name,
            as_of=as_of.isoformat(),
            peer_count=len(ranked),
            peers=ranked,
            valuation_leaders=_leaders(ranked, "valuation_score"),
            quality_leaders=_leaders(ranked, "quality_score"),
            growth_leaders=_leaders(ranked, "growth_score"),
            risk_leaders=_leaders(ranked, "risk_score"),
            composite_leaders=_leaders(ranked, "composite_score"),
            warnings=warnings,
        )

    def _security(self, symbol: str) -> SecurityMasterRecord | None:
        rows = [
            record
            for record in self.store.get_security_master([symbol])
            if record.exchange.upper() == self.venue
        ]
        return rows[0] if rows else None


def render_peer_markdown(report: PeerComparisonReport) -> str:
    lines = [
        f"# {report.symbol} Peer Comparison",
        "",
        f"Sector: {report.sector}",
        f"As of: {report.as_of}",
        f"Peer Count: {report.peer_count}",
        "",
    ]
    if report.target is not None:
        target = report.target
        lines.extend(
            [
                "## Target Ranks",
                f"- Composite: {target.composite_rank or 'unavailable'}",
                f"- Valuation: {target.valuation_rank or 'unavailable'}",
                f"- Quality: {target.quality_rank or 'unavailable'}",
                f"- Growth: {target.growth_rank or 'unavailable'}",
                f"- Risk: {target.risk_rank or 'unavailable'}",
                "",
                "## Thesis",
                *[f"- {item}" for item in report.thesis],
                "",
            ]
        )

    lines.extend(
        [
            "## Leaders",
            f"- Composite: {', '.join(report.composite_leaders) or 'unavailable'}",
            f"- Valuation: {', '.join(report.valuation_leaders) or 'unavailable'}",
            f"- Quality: {', '.join(report.quality_leaders) or 'unavailable'}",
            f"- Growth: {', '.join(report.growth_leaders) or 'unavailable'}",
            "",
            "## Peer Table",
            "| Rank | Symbol | Composite | Valuation | Quality | Growth | Risk | PE | PB | ROE | Rev Growth |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report.peers[:15]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.composite_rank or ""),
                    row.symbol,
                    _fmt(row.composite_score),
                    _fmt(row.valuation_score),
                    _fmt(row.quality_score),
                    _fmt(row.growth_score),
                    _fmt(row.risk_score),
                    _fmt(row.pe_ratio),
                    _fmt(row.pb_ratio),
                    _fmt(row.roe),
                    _fmt(row.revenue_growth_yoy),
                ]
            )
            + " |"
        )
    if report.warnings:
        lines.extend(["", "## Warnings", *[f"- {item}" for item in report.warnings]])
    return "\n".join(lines).strip() + "\n"


def render_sector_peer_markdown(report: SectorPeerComparisonReport) -> str:
    lines = [
        f"# {report.sector} Peer Rankings",
        "",
        f"As of: {report.as_of}",
        f"Peer Count: {report.peer_count}",
        "",
        "## Best Expressions",
        f"- Composite: {', '.join(report.composite_leaders) or 'unavailable'}",
        f"- Valuation: {', '.join(report.valuation_leaders) or 'unavailable'}",
        f"- Quality: {', '.join(report.quality_leaders) or 'unavailable'}",
        f"- Growth: {', '.join(report.growth_leaders) or 'unavailable'}",
        "",
        "## Peer Table",
        "| Rank | Symbol | Composite | Valuation | Quality | Growth | Risk | PE | PB |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.peers[:20]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.composite_rank or ""),
                    row.symbol,
                    _fmt(row.composite_score),
                    _fmt(row.valuation_score),
                    _fmt(row.quality_score),
                    _fmt(row.growth_score),
                    _fmt(row.risk_score),
                    _fmt(row.pe_ratio),
                    _fmt(row.pb_ratio),
                ]
            )
            + " |"
        )
    if report.warnings:
        lines.extend(["", "## Warnings", *[f"- {item}" for item in report.warnings]])
    return "\n".join(lines).strip() + "\n"


def _base_row(
    security: SecurityMasterRecord,
    fundamentals: FundamentalsSnapshot | None,
) -> PeerComparisonRow:
    if fundamentals is None:
        return PeerComparisonRow(
            symbol=security.symbol,
            company_name=security.company_name or security.symbol,
            sector=security.sector,
            industry=security.industry,
            missing_data=["fundamentals"],
        )
    pe = _positive(fundamentals.pe_ratio)
    pb = _positive(fundamentals.pb_ratio)
    earnings_yield = round(1.0 / pe, 4) if pe else None
    missing = []
    if pe is None:
        missing.append("pe_ratio")
    if pb is None:
        missing.append("pb_ratio")
    return PeerComparisonRow(
        symbol=security.symbol,
        company_name=security.company_name or security.symbol,
        sector=security.sector,
        industry=security.industry,
        pe_ratio=pe,
        pb_ratio=pb,
        earnings_yield=earnings_yield,
        revenue_growth_yoy=_finite(fundamentals.revenue_growth_yoy),
        earnings_growth_yoy=_finite(fundamentals.earnings_growth_yoy),
        roe=_finite(fundamentals.roe),
        roce=_finite(fundamentals.roce),
        operating_margin=_finite(fundamentals.operating_margin),
        net_profit_margin=_finite(fundamentals.net_profit_margin),
        free_cash_flow_margin=_finite(fundamentals.free_cash_flow_margin),
        debt_to_equity=_finite(fundamentals.debt_to_equity),
        interest_coverage=_positive(fundamentals.interest_coverage),
        missing_data=missing,
    )


def _score_rows(rows: Sequence[PeerComparisonRow]) -> list[PeerComparisonRow]:
    valuation_metrics = [
        _percentiles(rows, "pe_ratio", higher_is_better=False),
        _percentiles(rows, "pb_ratio", higher_is_better=False),
        _percentiles(rows, "earnings_yield", higher_is_better=True),
    ]
    quality_metrics = [
        _percentiles(rows, "roe", higher_is_better=True),
        _percentiles(rows, "roce", higher_is_better=True),
        _percentiles(rows, "operating_margin", higher_is_better=True),
        _percentiles(rows, "net_profit_margin", higher_is_better=True),
        _percentiles(rows, "free_cash_flow_margin", higher_is_better=True),
    ]
    growth_metrics = [
        _percentiles(rows, "revenue_growth_yoy", higher_is_better=True),
        _percentiles(rows, "earnings_growth_yoy", higher_is_better=True),
    ]
    risk_metrics = [
        _percentiles(rows, "debt_to_equity", higher_is_better=False),
        _percentiles(rows, "interest_coverage", higher_is_better=True),
    ]

    scored: list[PeerComparisonRow] = []
    for row in rows:
        valuation = _average_score(row.symbol, valuation_metrics)
        quality = _average_score(row.symbol, quality_metrics)
        growth = _average_score(row.symbol, growth_metrics)
        risk = _average_score(row.symbol, risk_metrics)
        composite = 0.25 * valuation + 0.30 * quality + 0.25 * growth + 0.20 * risk
        scored.append(
            replace(
                row,
                valuation_score=round(valuation, 2),
                quality_score=round(quality, 2),
                growth_score=round(growth, 2),
                risk_score=round(risk, 2),
                composite_score=round(composite, 2),
            )
        )

    scored = _assign_rank(scored, "valuation_score", "valuation_rank")
    scored = _assign_rank(scored, "quality_score", "quality_rank")
    scored = _assign_rank(scored, "growth_score", "growth_rank")
    scored = _assign_rank(scored, "risk_score", "risk_rank")
    return _assign_rank(scored, "composite_score", "composite_rank")


def _percentiles(
    rows: Sequence[PeerComparisonRow],
    field: str,
    higher_is_better: bool,
) -> dict[str, float]:
    values: list[tuple[str, float]] = []
    for row in rows:
        value = getattr(row, field)
        if value is None:
            continue
        values.append((row.symbol, float(value)))
    if not values:
        return {}
    values.sort(key=lambda item: item[1], reverse=higher_is_better)
    if len(values) == 1:
        return {values[0][0]: 100.0}
    out: dict[str, float] = {}
    denom = len(values) - 1
    for idx, (symbol, _value) in enumerate(values):
        out[symbol] = 100.0 * (1.0 - (idx / denom))
    return out


def _average_score(symbol: str, maps: Sequence[Mapping[str, float]]) -> float:
    values = [metric[symbol] for metric in maps if symbol in metric]
    return sum(values) / len(values) if values else 0.0


def _assign_rank(rows: Sequence[PeerComparisonRow], score_field: str, rank_field: str) -> list[PeerComparisonRow]:
    ranked = sorted(rows, key=lambda row: getattr(row, score_field), reverse=True)
    rank_by_symbol = {row.symbol: idx for idx, row in enumerate(ranked, start=1)}
    return [replace(cast(Any, row), **{rank_field: rank_by_symbol[row.symbol]}) for row in rows]


def _leaders(rows: Sequence[PeerComparisonRow], score_field: str, limit: int = 5) -> list[str]:
    ranked = sorted(rows, key=lambda row: getattr(row, score_field), reverse=True)
    return [row.symbol for row in ranked[:limit] if getattr(row, score_field) > 0.0]


def _target_thesis(target: PeerComparisonRow | None) -> list[str]:
    if target is None:
        return ["Peer comparison unavailable because target fundamentals are missing."]
    out: list[str] = []
    if target.composite_rank is not None and target.composite_rank <= 3:
        out.append("Ranks among the strongest composite peer expressions in the covered sector.")
    if target.valuation_rank is not None and target.valuation_rank <= 3:
        out.append("Valuation is favorable versus covered peers.")
    elif target.valuation_rank is not None and target.valuation_rank >= max(5, target.composite_rank or 0):
        out.append("Valuation is not the main support; quality or growth must justify the multiple.")
    if target.quality_rank is not None and target.quality_rank <= 3:
        out.append("Quality metrics rank well versus peers.")
    if target.growth_rank is not None and target.growth_rank <= 3:
        out.append("Growth metrics rank well versus peers.")
    if target.risk_rank is not None and target.risk_rank > 5:
        out.append("Balance-sheet risk metrics rank behind stronger peers.")
    if not out:
        out.append("Peer ranking is mixed; no dominant valuation, growth, or quality edge.")
    return out


def _positive(value: float) -> float | None:
    finite = _finite(value)
    if finite is None or finite <= 0.0:
        return None
    return finite


def _finite(value: float) -> float | None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return round(float(value), 4)


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"
