from __future__ import annotations

from datetime import date

from stock_screener_engine.data_sources.schemas import (
    BankingFactorRecord,
    EquityValuationRecord,
    FinancialStatementRecord,
    SecurityMasterRecord,
    ShareholdingRecord,
)
from stock_screener_engine.pipelines.factor_qa import CanonicalFactorQAReporter
from stock_screener_engine.storage.market_data_store import MarketDataStore


def test_factor_qa_reports_latest_factors_and_missing_sections(tmp_path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        store.upsert_security_master(
            [
                SecurityMasterRecord(symbol="AAA", sector="Consumer", industry="FMCG"),
                SecurityMasterRecord(symbol="BBB", sector="IT", industry="Services"),
            ]
        )
        store.upsert_financial_statements(
            [
                FinancialStatementRecord(
                    venue="NSE",
                    symbol="AAA",
                    period_end=date(2026, 3, 31),
                    filing_date=date(2026, 4, 20),
                    statement_type="annual",
                    revenue=1000.0,
                    ebit=200.0,
                    net_income=100.0,
                    operating_cash_flow=120.0,
                    capex=20.0,
                    total_debt=50.0,
                    equity=500.0,
                    total_assets=900.0,
                    current_assets=300.0,
                    current_liabilities=150.0,
                    interest_expense=10.0,
                    source_id="fy26",
                )
            ]
        )
        store.upsert_equity_valuations(
            [
                EquityValuationRecord(
                    venue="NSE",
                    symbol="AAA",
                    as_of=date(2026, 4, 30),
                    market_cap=2000.0,
                    shares_outstanding=100.0,
                    source_id="quote",
                )
            ]
        )
        store.upsert_shareholding(
            [
                ShareholdingRecord(
                    venue="NSE",
                    symbol="AAA",
                    period_end=date(2026, 3, 31),
                    filing_date=date(2026, 4, 15),
                    promoter_pct=50.0,
                    fii_pct=10.0,
                    dii_pct=15.0,
                    public_pct=25.0,
                    source_id="q4",
                )
            ]
        )

        report = CanonicalFactorQAReporter(store=store).build(
            symbols=["AAA", "BBB"],
            as_of=date(2026, 5, 1),
        )
    finally:
        store.close()

    assert report["passed"] is False
    assert report["summary"]["financials_available"] == 1
    assert report["summary"]["valuations_available"] == 1
    assert report["summary"]["shareholding_available"] == 1
    assert report["summary"]["error_count"] == 2

    aaa = report["symbols"][0]
    assert aaa["symbol"] == "AAA"
    assert aaa["status"] == "ok"
    assert aaa["derived_metrics"]["pe_ratio"] == 20.0
    assert aaa["derived_metrics"]["pb_ratio"] == 4.0
    assert aaa["valuation"]["implied_price"] == 20.0

    bbb = report["symbols"][1]
    assert bbb["status"] == "error"
    assert any(issue["section"] == "financials" for issue in bbb["quality_issues"])
    assert any(issue["section"] == "valuation" for issue in bbb["quality_issues"])


def test_factor_qa_uses_banking_factors_for_financial_business(tmp_path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        store.upsert_security_master(
            [
                SecurityMasterRecord(symbol="BANK", sector="Financial Services", industry="Banks"),
            ]
        )
        store.upsert_financial_statements(
            [
                FinancialStatementRecord(
                    venue="NSE",
                    symbol="BANK",
                    period_end=date(2026, 3, 31),
                    filing_date=date(2026, 4, 20),
                    statement_type="annual",
                    revenue=1000.0,
                    ebit=300.0,
                    net_income=150.0,
                    operating_cash_flow=180.0,
                    capex=10.0,
                    total_debt=500.0,
                    equity=1000.0,
                    total_assets=9000.0,
                    current_assets=0.0,
                    current_liabilities=0.0,
                    interest_expense=100.0,
                    source_id="fy26",
                )
            ]
        )
        store.upsert_equity_valuations(
            [
                EquityValuationRecord("NSE", "BANK", date(2026, 4, 30), market_cap=2500.0, shares_outstanding=100.0)
            ]
        )
        store.upsert_shareholding(
            [
                ShareholdingRecord("NSE", "BANK", date(2026, 3, 31), date(2026, 4, 20), 0.0, 45.0, 35.0, 20.0)
            ]
        )
        store.upsert_banking_factors(
            [
                BankingFactorRecord(
                    venue="NSE",
                    symbol="BANK",
                    period_end=date(2026, 3, 31),
                    filing_date=date(2026, 4, 20),
                    net_interest_income=500.0,
                    net_interest_margin_pct=3.8,
                    advances_growth_pct=12.0,
                    deposits_growth_pct=10.0,
                    casa_ratio_pct=40.0,
                    gnpa_ratio_pct=1.2,
                    nnpa_ratio_pct=0.4,
                    provision_coverage_ratio_pct=75.0,
                    credit_cost_pct=0.5,
                    capital_adequacy_ratio_pct=18.0,
                    cet1_ratio_pct=16.0,
                    cost_to_income_ratio_pct=38.0,
                    roa_pct=1.8,
                    roe_pct=15.0,
                    loan_to_deposit_ratio_pct=82.0,
                )
            ]
        )

        report = CanonicalFactorQAReporter(store=store).build(symbols=["BANK"], as_of=date(2026, 5, 1))
    finally:
        store.close()

    row = report["symbols"][0]
    assert row["status"] == "ok"
    assert row["banking"]["available"] is True
    assert row["banking_metrics"]["banking_quality_score"] > 70
    assert not any(issue["section"] == "financials" for issue in row["quality_issues"])
