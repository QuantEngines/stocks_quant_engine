from __future__ import annotations

from datetime import date

from stock_screener_engine.data_sources.financials.ingestion import FinancialStatementIngestor, PointInTimeStatementStore
from stock_screener_engine.monitoring.factor_quality import FactorQualityValidator


def test_ingestor_enforces_point_in_time() -> None:
    ingestor = FinancialStatementIngestor()
    as_of = date(2025, 3, 31)
    rows = [
        {
            "period_end": "2024-12-31",
            "filing_date": "2025-01-25",
            "statement_type": "quarterly",
            "revenue": 1000,
            "ebit": 170,
            "net_income": 120,
            "operating_cash_flow": 150,
            "capex": 40,
            "total_debt": 200,
            "equity": 500,
            "total_assets": 1200,
            "current_assets": 400,
            "current_liabilities": 220,
            "interest_expense": 30,
        },
        {
            "period_end": "2025-06-30",
            "filing_date": "2025-07-20",
            "statement_type": "quarterly",
            "revenue": 1200,
            "ebit": 200,
            "net_income": 140,
            "operating_cash_flow": 180,
            "capex": 45,
            "total_debt": 210,
            "equity": 520,
            "total_assets": 1300,
            "current_assets": 420,
            "current_liabilities": 240,
            "interest_expense": 32,
        },
    ]
    out = ingestor.ingest_rows(rows, venue="NSE", symbol="ABC", as_of=as_of)
    assert len(out.records) == 1
    assert out.rejected_rows == 1


def test_pit_store_returns_latest_eligible_record() -> None:
    ingestor = FinancialStatementIngestor()
    store = PointInTimeStatementStore()
    as_of = date(2025, 3, 31)
    rows = [
        {
            "period_end": "2024-09-30",
            "filing_date": "2024-10-20",
            "statement_type": "quarterly",
            "revenue": 900,
            "ebit": 150,
            "net_income": 110,
            "operating_cash_flow": 140,
            "capex": 35,
            "total_debt": 220,
            "equity": 470,
            "total_assets": 1150,
            "current_assets": 380,
            "current_liabilities": 200,
            "interest_expense": 28,
        },
        {
            "period_end": "2024-12-31",
            "filing_date": "2025-01-25",
            "statement_type": "quarterly",
            "revenue": 1000,
            "ebit": 170,
            "net_income": 120,
            "operating_cash_flow": 150,
            "capex": 40,
            "total_debt": 200,
            "equity": 500,
            "total_assets": 1200,
            "current_assets": 400,
            "current_liabilities": 220,
            "interest_expense": 30,
        },
    ]
    result = ingestor.ingest_rows(rows, venue="NSE", symbol="ABC", as_of=as_of)
    store.add(result.records)
    latest = store.latest_as_of("ABC", as_of)
    assert latest is not None
    assert latest.period_end.isoformat() == "2024-12-31"


def test_factor_quality_flags_errors() -> None:
    ingestor = FinancialStatementIngestor()
    validator = FactorQualityValidator()
    as_of = date(2025, 3, 31)
    rows = [
        {
            "period_end": "2024-12-31",
            "filing_date": "2025-01-25",
            "statement_type": "quarterly",
            "revenue": -1,
            "ebit": -20,
            "net_income": 10,
            "operating_cash_flow": 5,
            "capex": 2,
            "total_debt": 100,
            "equity": -5,
            "total_assets": 100,
            "current_assets": 120,
            "current_liabilities": 50,
            "interest_expense": 3,
        }
    ]
    result = ingestor.ingest_rows(rows, venue="NSE", symbol="ABC", as_of=as_of)
    report = validator.validate(result.records, as_of=as_of)
    assert not report.passed
    assert report.issues
