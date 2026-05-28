from __future__ import annotations

import json
from datetime import date, timedelta

from stock_screener_engine.data_sources.schemas import (
    BankingFactorRecord,
    EquityValuationRecord,
    FinancialStatementRecord,
    OHLCVBar,
    SecurityMasterRecord,
    ShareholdingRecord,
)
from stock_screener_engine.config.settings import DataSourceEntitlementSettings
from stock_screener_engine.pipelines.data_source_coverage import DataSourceCoverageReporter
from stock_screener_engine.storage.local_files import LocalFileStorage
from stock_screener_engine.storage.market_data_store import MarketDataStore


def test_data_source_coverage_report_aggregates_canonical_and_artifact_evidence(tmp_path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    storage = LocalFileStorage(str(tmp_path))
    as_of = date(2026, 5, 28)
    start = as_of - timedelta(days=5)
    store.upsert_security_master(
        [
            SecurityMasterRecord(symbol="AAA", company_name="AAA Ltd", sector="Financial Services", industry="Banks"),
            SecurityMasterRecord(symbol="BBB", company_name="BBB Ltd", sector="Information Technology", industry="IT Services"),
        ]
    )
    store.upsert_ohlcv(
        [
            OHLCVBar("NSE", "AAA", "2026-05-27", 100.0, 101.0, 99.0, 100.0, 1_000_000),
            OHLCVBar("NSE", "BBB", "2026-05-27", 200.0, 201.0, 199.0, 200.0, 1_000_000),
        ]
    )
    store.upsert_financial_statements(
        [
            FinancialStatementRecord(
                venue="NSE",
                symbol="AAA",
                period_end=date(2026, 3, 31),
                filing_date=date(2026, 5, 1),
                statement_type="annual_standalone",
                revenue=100.0,
                ebit=20.0,
                net_income=10.0,
                operating_cash_flow=12.0,
                capex=2.0,
                total_debt=30.0,
                equity=80.0,
                total_assets=150.0,
                current_assets=60.0,
                current_liabilities=20.0,
                interest_expense=1.0,
            )
        ]
    )
    store.upsert_equity_valuations(
        [EquityValuationRecord("NSE", "AAA", date(2026, 5, 27), market_cap=1_000.0, shares_outstanding=10.0)]
    )
    store.upsert_shareholding(
        [ShareholdingRecord("NSE", "AAA", date(2026, 3, 31), date(2026, 4, 20), 50.0, 20.0, 20.0, 10.0)]
    )
    store.upsert_banking_factors(
        [
            BankingFactorRecord(
                "NSE",
                "AAA",
                date(2026, 3, 31),
                date(2026, 5, 1),
                net_interest_margin_pct=3.2,
                gnpa_ratio_pct=1.2,
                nnpa_ratio_pct=0.4,
                cet1_ratio_pct=15.0,
                roa_pct=1.5,
                roe_pct=12.0,
            )
        ]
    )
    (tmp_path / "quality").mkdir(exist_ok=True)
    (tmp_path / "quality" / "broker_health_report.json").write_text(
        json.dumps(
            {
                "pipeline": "broker_health",
                "passed": True,
                "source_reports": {
                    "zerodha": {"enabled": True, "quote_coverage": 1.0, "historical_coverage": 1.0, "role": "primary"},
                    "icici_breeze": {"enabled": True, "quote_coverage": 0.5, "historical_coverage": 1.0, "role": "lagged_reconciliation"},
                },
            }
        ),
        encoding="utf-8",
    )
    finedge_root = tmp_path / "factors" / "finedge_trial"
    finedge_root.mkdir(parents=True)
    (finedge_root / "finedge_factor_export_report.json").write_text(
        json.dumps(
            {
                "pipeline": "finedge_factor_export",
                "passed": True,
                "per_symbol": {
                    "AAA": {"financial_rows": 1, "valuation_rows": 1, "shareholding_rows": 1},
                    "BBB": {"financial_rows": 0, "valuation_rows": 0, "shareholding_rows": 0},
                },
                "issues": [{"symbol": "BBB", "section": "financials", "message": "HTTP Error 401"}],
            }
        ),
        encoding="utf-8",
    )

    entitlements = [
        DataSourceEntitlementSettings(
            source_id="finedge",
            display_name="FinEdge",
            role="fundamentals",
            status="basic_sandbox",
            plan_name="Basic Free",
            domains=["financials", "valuations", "shareholding", "banking_factors"],
            allowed_symbols=["AAA"],
        )
    ]
    report = DataSourceCoverageReporter(store=store, file_store=storage, entitlements=entitlements).build(
        symbols=["AAA", "BBB"],
        as_of=as_of,
        start=start,
    )

    assert report["gross_coverage"]["market_data_coverage"] == 1.0
    assert report["gross_coverage"]["financial_statement_coverage"] == 0.5
    assert report["gross_coverage"]["banking_applicable_coverage"] == 1.0
    assert any(row["source"] == "FinEdge" and row["gross_coverage"] == 0.5 for row in report["sources"])
    financials = next(row for row in report["domains"] if row["domain"] == "financials")
    assert financials["entitlement"]["entitled_count"] == 1
    assert financials["entitlement"]["within_entitlement_coverage"] == 1.0
    assert "plan/access scope" in financials["entitlement_explanation"]
    finedge = next(row for row in report["sources"] if row["source"] == "FinEdge")
    assert finedge["entitlement"]["plan_name"] == "Basic Free"
    assert (tmp_path / "quality" / "data_source_coverage_report.json").exists()
    assert (tmp_path / "quality" / "data_source_coverage_report.md").exists()
    assert "Data Source Coverage" in str(report["markdown"])
    store.close()
