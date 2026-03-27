from __future__ import annotations

from datetime import date

from stock_screener_engine.data_sources.financials.provider import PointInTimeFinancialsProvider


def test_pit_financials_provider_builds_snapshots() -> None:
    provider = PointInTimeFinancialsProvider()
    rows = [
        {
            "period_end": "2024-12-31",
            "filing_date": "2025-01-20",
            "statement_type": "quarterly",
            "revenue": 1000,
            "ebit": 150,
            "net_income": 100,
            "operating_cash_flow": 140,
            "capex": 40,
            "total_debt": 200,
            "equity": 500,
            "total_assets": 1200,
            "current_assets": 400,
            "current_liabilities": 250,
            "interest_expense": 30,
        }
    ]
    summary = provider.ingest_statement_rows(
        venue="NSE",
        symbol="ABC",
        rows=rows,
        as_of=date(2025, 3, 31),
    )
    assert summary.accepted == 1

    f = provider.get_fundamentals(["ABC"])
    g = provider.get_governance(["ABC"])
    assert "ABC" in f
    assert "ABC" in g
    assert f["ABC"].debt_to_equity > 0
