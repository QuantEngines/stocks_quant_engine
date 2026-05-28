"""FinancialsProvider backed by canonical SQLite financial statements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

from stock_screener_engine.core.entities import FundamentalsSnapshot, GovernanceSnapshot
from stock_screener_engine.data_sources.base.interfaces import FinancialsProvider
from stock_screener_engine.data_sources.schemas import (
    BankingFactorRecord,
    EquityValuationRecord,
    FinancialStatementRecord,
    ShareholdingRecord,
)
from stock_screener_engine.storage.market_data_store import MarketDataStore


class SQLiteFinancialsProvider(FinancialsProvider):
    """Build point-in-time fundamental snapshots from stored statements.

    It prefers the latest four eligible quarterly statements for TTM metrics.
    If four quarters are unavailable, it falls back to the latest eligible
    annual statement. PE/PB are computed only when a canonical market-cap record
    exists on or before the requested as-of date.
    """

    def __init__(
        self,
        sqlite_path: str,
        venue: str = "NSE",
        store: MarketDataStore | None = None,
    ) -> None:
        self.store = store or MarketDataStore(sqlite_path)
        self.venue = venue.strip().upper() or "NSE"

    def get_fundamentals(self, symbols: Sequence[str]) -> dict[str, FundamentalsSnapshot]:
        return self.get_fundamentals_as_of(symbols, as_of=date.today())

    def get_fundamentals_as_of(
        self,
        symbols: Sequence[str],
        as_of: date,
    ) -> dict[str, FundamentalsSnapshot]:
        out: dict[str, FundamentalsSnapshot] = {}
        for symbol in _normalise_symbols(symbols):
            records = self.store.get_financial_statements(symbol=symbol, venue=self.venue, as_of=as_of)
            if not records:
                continue
            current = _build_current_ttm(records)
            if current is None:
                continue
            comparison = _build_prior_ttm(records, current)
            valuation = self.store.latest_equity_valuation_as_of(
                symbol=symbol,
                as_of=as_of,
                venue=self.venue,
            )
            out[symbol] = _fundamentals_from_ttm(current, comparison, valuation)
        return out

    def get_peer_context_as_of(
        self,
        sectors: Sequence[str],
        as_of: date,
    ) -> dict[str, dict[str, object]]:
        """Return same-sector active securities with point-in-time fundamentals.

        The research engine uses this to compute sector-relative valuation
        context from the full canonical peer set, even when a scan is running
        over a small symbol subset.
        """
        securities = self.store.list_active_securities(sectors=sectors, exchange=self.venue)
        symbols = [security.symbol for security in securities]
        fundamentals = self.get_fundamentals_as_of(symbols, as_of=as_of)
        out: dict[str, dict[str, object]] = {}
        for security in securities:
            snapshot = fundamentals.get(security.symbol)
            if snapshot is None:
                continue
            out[security.symbol] = {
                "symbol": security.symbol,
                "company_name": security.company_name,
                "sector": security.sector,
                "industry": security.industry,
                "fundamentals": snapshot,
            }
        return out

    def get_governance(self, symbols: Sequence[str]) -> dict[str, GovernanceSnapshot]:
        return self.get_governance_as_of(symbols, as_of=date.today())

    def get_governance_as_of(
        self,
        symbols: Sequence[str],
        as_of: date,
    ) -> dict[str, GovernanceSnapshot]:
        out: dict[str, GovernanceSnapshot] = {}
        for symbol in _normalise_symbols(symbols):
            records = self.store.get_shareholding(symbol=symbol, venue=self.venue, as_of=as_of)
            if not records:
                continue
            out[symbol] = _governance_from_shareholding(records[0], records[1] if len(records) > 1 else None)
        return out

    def get_banking_factors_as_of(
        self,
        symbols: Sequence[str],
        as_of: date,
    ) -> dict[str, BankingFactorRecord]:
        out: dict[str, BankingFactorRecord] = {}
        for symbol in _normalise_symbols(symbols):
            record = self.store.latest_banking_factor_as_of(symbol=symbol, as_of=as_of, venue=self.venue)
            if record is not None:
                out[symbol] = record
        return out

    def coverage_report(self, symbols: Sequence[str], as_of: date) -> dict[str, object]:
        statement_coverage = self.store.financial_statement_coverage(
            symbols=symbols,
            as_of=as_of,
            venue=self.venue,
        )
        valuation_coverage = self.store.equity_valuation_coverage(
            symbols=symbols,
            as_of=as_of,
            venue=self.venue,
        )
        shareholding_coverage = self.store.shareholding_coverage(
            symbols=symbols,
            as_of=as_of,
            venue=self.venue,
        )
        banking_coverage = self.store.banking_factor_coverage(
            symbols=symbols,
            as_of=as_of,
            venue=self.venue,
        )
        return {
            "financial_statements": statement_coverage,
            "equity_valuations": valuation_coverage,
            "shareholding": shareholding_coverage,
            "banking_factors": banking_coverage,
        }

    def close(self) -> None:
        self.store.close()


@dataclass(frozen=True)
class TTMFinancials:
    symbol: str
    period_end: date
    revenue: float
    ebit: float
    net_income: float
    operating_cash_flow: float
    capex: float
    interest_expense: float
    total_debt: float
    equity: float
    total_assets: float
    current_assets: float
    current_liabilities: float
    source: str
    record_count: int


def _fundamentals_from_ttm(
    current: TTMFinancials,
    comparison: TTMFinancials | None,
    valuation: EquityValuationRecord | None,
) -> FundamentalsSnapshot:
    revenue = current.revenue
    invested_capital = current.total_assets - current.current_liabilities
    return FundamentalsSnapshot(
        symbol=current.symbol,
        as_of=current.period_end,
        pe_ratio=_valuation_ratio(valuation.market_cap if valuation else 0.0, current.net_income),
        pb_ratio=_valuation_ratio(valuation.market_cap if valuation else 0.0, current.equity),
        roe=_safe_div(current.net_income, current.equity),
        roa=_safe_div(current.net_income, current.total_assets),
        roce=_safe_div(current.ebit, invested_capital),
        debt_to_equity=max(0.0, _safe_div(current.total_debt, current.equity)),
        current_ratio=_safe_div(current.current_assets, current.current_liabilities),
        interest_coverage=_safe_div(current.ebit, current.interest_expense),
        earnings_growth_yoy=_growth(current.net_income, comparison.net_income if comparison else None),
        revenue_growth_yoy=_growth(current.revenue, comparison.revenue if comparison else None),
        free_cash_flow_margin=_safe_div(current.operating_cash_flow - current.capex, revenue),
        operating_margin=_safe_div(current.ebit, revenue),
        net_profit_margin=_safe_div(current.net_income, revenue),
    )


def _governance_from_shareholding(
    latest: ShareholdingRecord,
    previous: ShareholdingRecord | None,
) -> GovernanceSnapshot:
    promoter_change = (
        latest.promoter_pct - previous.promoter_pct
        if previous is not None
        else 0.0
    )
    institutional = latest.fii_pct + latest.dii_pct
    ownership_quality = _clamp01(
        0.45 * _clamp01(latest.promoter_pct / 55.0)
        + 0.35 * _clamp01(institutional / 35.0)
        + 0.20 * _clamp01((promoter_change + 2.0) / 4.0)
    )
    return GovernanceSnapshot(
        symbol=latest.symbol,
        as_of=latest.period_end,
        promoter_holding_pct=latest.promoter_pct,
        promoter_holding_change_qoq=promoter_change / 100.0,
        institutional_holding_pct=institutional,
        fii_holding_pct=latest.fii_pct,
        dii_holding_pct=latest.dii_pct,
        insider_activity_score=(ownership_quality * 2.0) - 1.0,
        audit_opinion="unknown",
    )


def _build_current_ttm(records: Sequence[FinancialStatementRecord]) -> TTMFinancials | None:
    annuals = [record for record in records if "annual" in record.statement_type.lower()]
    quarters = [record for record in records if "quarter" in record.statement_type.lower()]
    latest_annual = annuals[0] if annuals else None
    if len(quarters) >= 4 and (latest_annual is None or quarters[0].period_end >= latest_annual.period_end):
        return _aggregate_quarters(quarters[:4], source="ttm_quarterly")
    if latest_annual is not None:
        return _from_single_statement(latest_annual, source="annual")
    if len(quarters) >= 4:
        return _aggregate_quarters(quarters[:4], source="ttm_quarterly")
    return None


def _build_prior_ttm(
    records: Sequence[FinancialStatementRecord],
    current: TTMFinancials,
) -> TTMFinancials | None:
    cutoff = current.period_end - timedelta(days=300)
    if current.source == "ttm_quarterly":
        quarters = [
            record
            for record in records
            if "quarter" in record.statement_type.lower() and record.period_end <= cutoff
        ]
        if len(quarters) >= 4:
            return _aggregate_quarters(quarters[:4], source="ttm_quarterly")
        return None
    annuals = [
        record
        for record in records
        if "annual" in record.statement_type.lower() and record.period_end <= cutoff
    ]
    return _from_single_statement(annuals[0], source="annual") if annuals else None


def _aggregate_quarters(records: Sequence[FinancialStatementRecord], source: str) -> TTMFinancials:
    latest = records[0]
    return TTMFinancials(
        symbol=latest.symbol,
        period_end=latest.period_end,
        revenue=sum(record.revenue for record in records),
        ebit=sum(record.ebit for record in records),
        net_income=sum(record.net_income for record in records),
        operating_cash_flow=sum(record.operating_cash_flow for record in records),
        capex=sum(record.capex for record in records),
        interest_expense=sum(record.interest_expense for record in records),
        total_debt=latest.total_debt,
        equity=latest.equity,
        total_assets=latest.total_assets,
        current_assets=latest.current_assets,
        current_liabilities=latest.current_liabilities,
        source=source,
        record_count=len(records),
    )


def _from_single_statement(record: FinancialStatementRecord, source: str) -> TTMFinancials:
    return TTMFinancials(
        symbol=record.symbol,
        period_end=record.period_end,
        revenue=record.revenue,
        ebit=record.ebit,
        net_income=record.net_income,
        operating_cash_flow=record.operating_cash_flow,
        capex=record.capex,
        interest_expense=record.interest_expense,
        total_debt=record.total_debt,
        equity=record.equity,
        total_assets=record.total_assets,
        current_assets=record.current_assets,
        current_liabilities=record.current_liabilities,
        source=source,
        record_count=1,
    )


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-9:
        return 0.0
    return numerator / denominator


def _growth(current: float, previous: float | None) -> float:
    if previous is None or abs(previous) < 1e-9:
        return 0.0
    return (current - previous) / abs(previous)


def _valuation_ratio(market_cap: float, denominator: float) -> float:
    if market_cap <= 0 or denominator <= 0:
        return 0.0
    return market_cap / denominator


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalise_symbols(symbols: Sequence[str]) -> list[str]:
    return [symbol.strip().upper() for symbol in symbols if symbol and symbol.strip()]
