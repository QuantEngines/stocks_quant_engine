"""Normalized ingestion schema contracts.

These records define canonical, adapter-agnostic shapes for market/exchange/
financial ingestion data. Adapters should normalize source-specific fields into
these dataclasses before passing data into the core pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class OHLCVBar:
    venue: str
    symbol: str
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class CorporateActionRecord:
    venue: str
    symbol: str
    action_type: str
    ex_date: str
    record_date: str | None = None
    ratio: str | None = None
    cash_amount: float | None = None
    currency: str = "INR"
    source_id: str = ""


@dataclass(frozen=True)
class ShareholdingRecord:
    venue: str
    symbol: str
    period_end: date
    filing_date: date
    promoter_pct: float
    fii_pct: float
    dii_pct: float
    public_pct: float
    source_id: str = ""


@dataclass(frozen=True)
class AnnouncementRecord:
    venue: str
    symbol: str
    published_at: str
    category: str
    subject: str
    url: str
    source_id: str = ""


@dataclass(frozen=True)
class FinancialStatementRecord:
    venue: str
    symbol: str
    period_end: date
    filing_date: date
    statement_type: str
    revenue: float
    ebit: float
    net_income: float
    operating_cash_flow: float
    capex: float
    total_debt: float
    equity: float
    total_assets: float
    current_assets: float
    current_liabilities: float
    interest_expense: float
    source_id: str = ""


@dataclass(frozen=True)
class FactorQualityIssue:
    symbol: str
    as_of: date
    severity: str
    message: str
