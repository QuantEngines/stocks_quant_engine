"""Financial statement ingestion with point-in-time semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from stock_screener_engine.data_sources.schemas import FinancialStatementRecord


@dataclass(frozen=True)
class FinancialIngestionResult:
    records: list[FinancialStatementRecord]
    rejected_rows: int


class FinancialStatementIngestor:
    """Parses statement rows and enforces point-in-time eligibility.

    PIT rule:
    - statement `period_end` must be <= `as_of`
    - statement `filing_date` must be <= `as_of`
    """

    def ingest_rows(
        self,
        rows: list[dict],
        venue: str,
        symbol: str,
        as_of: date,
    ) -> FinancialIngestionResult:
        accepted: list[FinancialStatementRecord] = []
        rejected = 0
        for row in rows:
            try:
                rec = FinancialStatementRecord(
                    venue=venue,
                    symbol=symbol,
                    period_end=_parse_date(row.get("period_end")),
                    filing_date=_parse_date(row.get("filing_date")),
                    statement_type=str(row.get("statement_type", "quarterly")).lower(),
                    revenue=_f(row.get("revenue")),
                    ebit=_f(row.get("ebit")),
                    net_income=_f(row.get("net_income")),
                    operating_cash_flow=_f(row.get("operating_cash_flow")),
                    capex=_f(row.get("capex")),
                    total_debt=_f(row.get("total_debt")),
                    equity=_f(row.get("equity")),
                    total_assets=_f(row.get("total_assets")),
                    current_assets=_f(row.get("current_assets")),
                    current_liabilities=_f(row.get("current_liabilities")),
                    interest_expense=_f(row.get("interest_expense")),
                    source_id=str(row.get("source_id", "")),
                )
            except (TypeError, ValueError):
                rejected += 1
                continue

            if rec.period_end <= as_of and rec.filing_date <= as_of:
                accepted.append(rec)
            else:
                rejected += 1

        accepted.sort(key=lambda r: (r.period_end, r.filing_date), reverse=True)
        return FinancialIngestionResult(records=accepted, rejected_rows=rejected)


class PointInTimeStatementStore:
    """Simple in-memory PIT store used by research pipeline and tests."""

    def __init__(self) -> None:
        self._records: dict[str, list[FinancialStatementRecord]] = {}

    def add(self, records: list[FinancialStatementRecord]) -> None:
        for rec in records:
            self._records.setdefault(rec.symbol, []).append(rec)
        for symbol in list(self._records.keys()):
            self._records[symbol].sort(key=lambda r: (r.period_end, r.filing_date), reverse=True)

    def latest_as_of(self, symbol: str, as_of: date) -> FinancialStatementRecord | None:
        for rec in self._records.get(symbol, []):
            if rec.period_end <= as_of and rec.filing_date <= as_of:
                return rec
        return None


def _parse_date(value: object) -> date:
    if isinstance(value, date):
        return value
    if value is None:
        raise ValueError("date missing")
    text = str(value).strip()
    if not text:
        raise ValueError("date missing")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"unsupported date format: {text}") from exc


def _f(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value).strip())
    except ValueError:
        return 0.0
