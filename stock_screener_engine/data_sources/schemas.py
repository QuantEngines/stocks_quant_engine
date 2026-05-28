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
class DeliveryTurnoverRecord:
    venue: str
    symbol: str
    trade_date: date
    traded_quantity: float
    delivery_quantity: float
    delivery_pct: float
    source_id: str = ""


@dataclass(frozen=True)
class SecurityMasterRecord:
    symbol: str
    isin: str = ""
    exchange: str = "NSE"
    series: str = "EQ"
    company_name: str = ""
    sector: str = "Unknown"
    industry: str = "Unknown"
    listing_date: date | None = None
    delisting_date: date | None = None
    active: bool = True
    lot_size: int = 1
    tick_size: float = 0.05
    source: str = "manual"


@dataclass(frozen=True)
class MarketSessionRecord:
    venue: str
    session_date: date
    is_trading_day: bool
    session_type: str = "regular"
    reason: str = ""


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
class BankingFactorRecord:
    venue: str
    symbol: str
    period_end: date
    filing_date: date
    net_interest_income: float = 0.0
    net_interest_margin_pct: float = 0.0
    advances_growth_pct: float = 0.0
    deposits_growth_pct: float = 0.0
    casa_ratio_pct: float = 0.0
    gnpa_ratio_pct: float = 0.0
    nnpa_ratio_pct: float = 0.0
    provision_coverage_ratio_pct: float = 0.0
    credit_cost_pct: float = 0.0
    capital_adequacy_ratio_pct: float = 0.0
    cet1_ratio_pct: float = 0.0
    cost_to_income_ratio_pct: float = 0.0
    roa_pct: float = 0.0
    roe_pct: float = 0.0
    loan_to_deposit_ratio_pct: float = 0.0
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
class EquityValuationRecord:
    venue: str
    symbol: str
    as_of: date
    market_cap: float
    shares_outstanding: float = 0.0
    free_float_market_cap: float = 0.0
    enterprise_value: float = 0.0
    currency: str = "INR"
    source_id: str = ""


@dataclass(frozen=True)
class FactorQualityIssue:
    symbol: str
    as_of: date
    severity: str
    message: str
