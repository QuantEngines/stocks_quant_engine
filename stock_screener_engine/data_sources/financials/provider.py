"""FinancialsProvider implementation backed by point-in-time statements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

from stock_screener_engine.core.entities import FundamentalsSnapshot, GovernanceSnapshot
from stock_screener_engine.data_sources.base.interfaces import FinancialsProvider
from stock_screener_engine.data_sources.financials.ingestion import (
    FinancialStatementIngestor,
    PointInTimeStatementStore,
)
from stock_screener_engine.data_sources.schemas import FinancialStatementRecord
from stock_screener_engine.monitoring.factor_quality import FactorQualityValidator


@dataclass(frozen=True)
class IngestionSummary:
    accepted: int
    rejected: int
    quality_issues: int


class PointInTimeFinancialsProvider(FinancialsProvider):
    """Builds snapshots from statement records while preserving PIT constraints."""

    def __init__(self) -> None:
        self._store = PointInTimeStatementStore()
        self._validator = FactorQualityValidator()

    def ingest_statement_rows(
        self,
        venue: str,
        symbol: str,
        rows: list[dict],
        as_of: date,
    ) -> IngestionSummary:
        ingestor = FinancialStatementIngestor()
        ingested = ingestor.ingest_rows(rows=rows, venue=venue, symbol=symbol, as_of=as_of)
        quality = self._validator.validate(ingested.records, as_of=as_of)
        self._store.add(ingested.records)
        return IngestionSummary(
            accepted=len(ingested.records),
            rejected=ingested.rejected_rows,
            quality_issues=len(quality.issues),
        )

    def ingest_records(self, records: list[FinancialStatementRecord]) -> None:
        self._store.add(records)

    def get_fundamentals(self, symbols: Sequence[str]) -> dict[str, FundamentalsSnapshot]:
        today = date.today()
        out: dict[str, FundamentalsSnapshot] = {}
        for symbol in symbols:
            rec = self._store.latest_as_of(symbol, today)
            if rec is None:
                continue
            equity = rec.equity if abs(rec.equity) > 1e-9 else 1.0
            assets = rec.total_assets if abs(rec.total_assets) > 1e-9 else 1.0
            revenue = rec.revenue if abs(rec.revenue) > 1e-9 else 1.0
            debt_to_equity = max(0.0, rec.total_debt / abs(equity))
            current_ratio = rec.current_assets / max(1e-9, rec.current_liabilities)
            interest_coverage = rec.ebit / max(1e-9, rec.interest_expense)
            out[symbol] = FundamentalsSnapshot(
                symbol=symbol,
                as_of=rec.period_end,
                pe_ratio=0.0,
                pb_ratio=0.0,
                roe=rec.net_income / equity,
                roa=rec.net_income / assets,
                roce=rec.ebit / max(1e-9, assets - rec.current_liabilities),
                debt_to_equity=debt_to_equity,
                current_ratio=current_ratio,
                interest_coverage=interest_coverage,
                earnings_growth_yoy=0.0,
                revenue_growth_yoy=0.0,
                free_cash_flow_margin=(rec.operating_cash_flow - rec.capex) / revenue,
                operating_margin=rec.ebit / revenue,
                net_profit_margin=rec.net_income / revenue,
            )
        return out

    def get_governance(self, symbols: Sequence[str]) -> dict[str, GovernanceSnapshot]:
        today = date.today()
        out: dict[str, GovernanceSnapshot] = {}
        for symbol in symbols:
            rec = self._store.latest_as_of(symbol, today)
            if rec is None:
                continue
            leverage_score = 1.0 - min(2.0, rec.total_debt / max(1e-9, abs(rec.equity))) / 2.0
            cash_quality = min(2.0, max(-2.0, (rec.operating_cash_flow - rec.capex) / max(1e-9, rec.revenue)))
            out[symbol] = GovernanceSnapshot(
                symbol=symbol,
                as_of=rec.period_end,
                promoter_holding_pct=0.0,
                promoter_holding_change_qoq=0.0,
                institutional_holding_pct=0.0,
                fii_holding_pct=0.0,
                dii_holding_pct=0.0,
                insider_activity_score=max(-1.0, min(1.0, 0.5 * leverage_score + 0.5 * cash_quality)),
                audit_opinion="clean",
            )
        return out
