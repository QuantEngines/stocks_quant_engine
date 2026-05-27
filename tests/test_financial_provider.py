from __future__ import annotations

from datetime import date

from stock_screener_engine.app import (
    run_factor_ingest,
    run_factor_template,
    run_financials_ingest,
    run_security_master_ingest,
    run_shareholding_ingest,
    run_valuation_ingest,
)
from stock_screener_engine.data_sources.financials.provider import PointInTimeFinancialsProvider
from stock_screener_engine.data_sources.financials.sqlite_financials_provider import SQLiteFinancialsProvider
from stock_screener_engine.data_sources.schemas import (
    EquityValuationRecord,
    FinancialStatementRecord,
    SecurityMasterRecord,
    ShareholdingRecord,
)
from stock_screener_engine.storage.market_data_store import MarketDataStore


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


def test_sqlite_financials_provider_builds_point_in_time_ratios(tmp_path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        store.upsert_financial_statements(
            [
                FinancialStatementRecord(
                    venue="NSE",
                    symbol="ABC",
                    period_end=date(2025, 3, 31),
                    filing_date=date(2025, 4, 15),
                    statement_type="annual",
                    revenue=1000.0,
                    ebit=180.0,
                    net_income=120.0,
                    operating_cash_flow=160.0,
                    capex=40.0,
                    total_debt=250.0,
                    equity=600.0,
                    total_assets=1500.0,
                    current_assets=500.0,
                    current_liabilities=300.0,
                    interest_expense=30.0,
                    source_id="fy25",
                ),
                FinancialStatementRecord(
                    venue="NSE",
                    symbol="ABC",
                    period_end=date(2026, 3, 31),
                    filing_date=date(2026, 4, 15),
                    statement_type="annual",
                    revenue=1250.0,
                    ebit=250.0,
                    net_income=180.0,
                    operating_cash_flow=240.0,
                    capex=50.0,
                    total_debt=200.0,
                    equity=720.0,
                    total_assets=1700.0,
                    current_assets=650.0,
                    current_liabilities=340.0,
                    interest_expense=25.0,
                    source_id="fy26",
                ),
            ]
        )
        store.upsert_shareholding(
            [
                ShareholdingRecord(
                    venue="NSE",
                    symbol="ABC",
                    period_end=date(2025, 12, 31),
                    filing_date=date(2026, 1, 20),
                    promoter_pct=50.0,
                    fii_pct=10.0,
                    dii_pct=15.0,
                    public_pct=25.0,
                    source_id="q3",
                ),
                ShareholdingRecord(
                    venue="NSE",
                    symbol="ABC",
                    period_end=date(2026, 3, 31),
                    filing_date=date(2026, 4, 15),
                    promoter_pct=52.0,
                    fii_pct=11.0,
                    dii_pct=16.0,
                    public_pct=21.0,
                    source_id="q4",
                ),
            ]
        )
    finally:
        store.close()

    provider = SQLiteFinancialsProvider(sqlite_path=str(tmp_path / "market.db"), venue="NSE")
    try:
        fundamentals = provider.get_fundamentals_as_of(["ABC"], as_of=date(2026, 5, 1))
        governance = provider.get_governance_as_of(["ABC"], as_of=date(2026, 5, 1))
    finally:
        provider.close()

    assert fundamentals["ABC"].revenue_growth_yoy == 0.25
    assert fundamentals["ABC"].earnings_growth_yoy == 0.5
    assert round(fundamentals["ABC"].roe, 4) == 0.25
    assert fundamentals["ABC"].pe_ratio == 0.0
    assert governance["ABC"].audit_opinion == "unknown"
    assert governance["ABC"].promoter_holding_pct == 52.0
    assert governance["ABC"].promoter_holding_change_qoq == 0.02


def test_sqlite_financials_provider_prefers_quarterly_ttm_and_market_cap(tmp_path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        store.upsert_financial_statements(
            [
                _statement("ABC", "2024-06-30", "2024-07-20", 200.0, 20.0, source_id="q1p"),
                _statement("ABC", "2024-09-30", "2024-10-20", 210.0, 21.0, source_id="q2p"),
                _statement("ABC", "2024-12-31", "2025-01-20", 220.0, 23.0, source_id="q3p"),
                _statement("ABC", "2025-03-31", "2025-04-20", 218.0, 24.0, source_id="q4p"),
                _statement("ABC", "2025-06-30", "2025-07-20", 250.0, 30.0, source_id="q1"),
                _statement("ABC", "2025-09-30", "2025-10-20", 260.0, 32.0, source_id="q2"),
                _statement("ABC", "2025-12-31", "2026-01-20", 270.0, 34.0, source_id="q3"),
                _statement("ABC", "2026-03-31", "2026-04-20", 280.0, 36.0, source_id="q4"),
            ]
        )
        store.upsert_equity_valuations(
            [
                EquityValuationRecord(
                    venue="NSE",
                    symbol="ABC",
                    as_of=date(2026, 4, 30),
                    market_cap=13200.0,
                    shares_outstanding=100.0,
                    source_id="mcap",
                )
            ]
        )
    finally:
        store.close()

    provider = SQLiteFinancialsProvider(sqlite_path=str(tmp_path / "market.db"), venue="NSE")
    try:
        fundamentals = provider.get_fundamentals_as_of(["ABC"], as_of=date(2026, 5, 1))
    finally:
        provider.close()

    assert fundamentals["ABC"].revenue_growth_yoy == 0.25
    assert fundamentals["ABC"].earnings_growth_yoy == 0.5
    assert fundamentals["ABC"].pe_ratio == 100.0
    assert fundamentals["ABC"].pb_ratio == 20.0


def test_sqlite_financials_provider_returns_sector_peer_context(tmp_path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        store.upsert_security_master(
            [
                SecurityMasterRecord(symbol="AAA", sector="IT", industry="Software"),
                SecurityMasterRecord(symbol="BBB", sector="IT", industry="Services"),
                SecurityMasterRecord(symbol="CCC", sector="IT", active=False),
                SecurityMasterRecord(symbol="DDD", sector="Banks"),
            ]
        )
        store.upsert_financial_statements(
            [
                _annual_statement("AAA", net_income=100.0, equity=500.0),
                _annual_statement("BBB", net_income=200.0, equity=1000.0),
                _annual_statement("CCC", net_income=150.0, equity=750.0),
                _annual_statement("DDD", net_income=300.0, equity=1200.0),
            ]
        )
        store.upsert_equity_valuations(
            [
                EquityValuationRecord("NSE", "AAA", date(2026, 4, 30), market_cap=2000.0),
                EquityValuationRecord("NSE", "BBB", date(2026, 4, 30), market_cap=3000.0),
                EquityValuationRecord("NSE", "CCC", date(2026, 4, 30), market_cap=4500.0),
                EquityValuationRecord("NSE", "DDD", date(2026, 4, 30), market_cap=6000.0),
            ]
        )
    finally:
        store.close()

    provider = SQLiteFinancialsProvider(sqlite_path=str(tmp_path / "market.db"), venue="NSE")
    try:
        peers = provider.get_peer_context_as_of(["it"], as_of=date(2026, 5, 1))
    finally:
        provider.close()

    assert set(peers) == {"AAA", "BBB"}
    assert peers["AAA"]["sector"] == "IT"
    assert peers["AAA"]["fundamentals"].pe_ratio == 20.0


def test_run_financials_ingest_persists_csv_rows(monkeypatch, tmp_path) -> None:
    csv_path = tmp_path / "statements.csv"
    csv_path.write_text(
        "\n".join(
            [
                "period_end,filing_date,statement_type,revenue,ebit,net_income,operating_cash_flow,capex,total_debt,equity,total_assets,current_assets,current_liabilities,interest_expense,source_id",
                "2026-03-31,2026-04-20,annual,1200,250,180,240,50,200,720,1700,650,340,25,fy26",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SSE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("SSE_SQLITE_PATH", str(tmp_path / "market.db"))

    report = run_financials_ingest(
        symbol="ABC",
        file_path=str(csv_path),
        as_of=date(2026, 5, 1),
    )

    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        rows = store.get_financial_statements("ABC", as_of=date(2026, 5, 1), venue="NSE")
    finally:
        store.close()

    assert report["passed"] is True
    assert report["persisted"] == 1
    assert rows[0].net_income == 180.0


def test_run_valuation_ingest_persists_csv_rows(monkeypatch, tmp_path) -> None:
    csv_path = tmp_path / "valuations.csv"
    csv_path.write_text(
        "\n".join(
            [
                "as_of,market_cap,shares_outstanding,free_float_market_cap,enterprise_value,currency,source_id",
                "2026-04-30,13200,100,9000,14000,INR,mcap",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SSE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("SSE_SQLITE_PATH", str(tmp_path / "market.db"))

    report = run_valuation_ingest(
        symbol="ABC",
        file_path=str(csv_path),
        as_of=date(2026, 5, 1),
    )

    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        rows = store.get_equity_valuations("ABC", as_of=date(2026, 5, 1), venue="NSE")
    finally:
        store.close()

    assert report["passed"] is True
    assert report["persisted"] == 1
    assert rows[0].market_cap == 13200.0


def test_run_shareholding_ingest_persists_csv_rows(monkeypatch, tmp_path) -> None:
    csv_path = tmp_path / "shareholding.csv"
    csv_path.write_text(
        "\n".join(
            [
                "period_end,filing_date,promoter_pct,fii_pct,dii_pct,public_pct,source_id",
                "2026-03-31,2026-04-20,52,11,16,21,q4",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SSE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("SSE_SQLITE_PATH", str(tmp_path / "market.db"))

    report = run_shareholding_ingest(
        symbol="ABC",
        file_path=str(csv_path),
        as_of=date(2026, 5, 1),
    )

    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        rows = store.get_shareholding("ABC", as_of=date(2026, 5, 1), venue="NSE")
    finally:
        store.close()

    assert report["passed"] is True
    assert report["persisted"] == 1
    assert rows[0].promoter_pct == 52.0


def test_factor_template_creates_external_bulk_csvs(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SSE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("SSE_SQLITE_PATH", str(tmp_path / "market.db"))
    output_root = tmp_path / "factors" / "nifty50"

    report = run_factor_template(
        output_root=str(output_root),
        symbols=["AAA", "BBB"],
        as_of=date(2026, 5, 1),
    )

    assert report["symbols"] == 2
    assert (output_root / "financials.csv").exists()
    assert (output_root / "valuations.csv").exists()
    assert (output_root / "shareholding.csv").exists()
    assert "AAA" in (output_root / "financials.csv").read_text(encoding="utf-8")


def test_factor_ingest_persists_bulk_pit_factors(monkeypatch, tmp_path) -> None:
    factor_root = tmp_path / "factor_input"
    factor_root.mkdir()
    (factor_root / "financials.csv").write_text(
        "\n".join(
            [
                "symbol,period_end,filing_date,statement_type,revenue,ebit,net_income,operating_cash_flow,capex,total_debt,equity,total_assets,current_assets,current_liabilities,interest_expense,source_id",
                "AAA,2026-03-31,2026-04-20,annual,1200,250,180,240,50,200,720,1700,650,340,25,fy26",
                "BBB,2026-03-31,2026-04-20,annual,900,160,110,130,30,180,500,1200,420,250,18,fy26",
            ]
        ),
        encoding="utf-8",
    )
    (factor_root / "valuations.csv").write_text(
        "\n".join(
            [
                "symbol,as_of,market_cap,shares_outstanding,free_float_market_cap,enterprise_value,currency,source_id",
                "AAA,2026-04-30,13200,100,9000,14000,INR,mcap",
                "BBB,2026-04-30,8800,100,6000,9000,INR,mcap",
            ]
        ),
        encoding="utf-8",
    )
    (factor_root / "shareholding.csv").write_text(
        "\n".join(
            [
                "symbol,period_end,filing_date,promoter_pct,fii_pct,dii_pct,public_pct,source_id",
                "AAA,2026-03-31,2026-04-20,52,11,16,21,q4",
                "BBB,2026-03-31,2026-04-20,45,15,20,20,q4",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SSE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("SSE_SQLITE_PATH", str(tmp_path / "market.db"))

    report = run_factor_ingest(
        root=str(factor_root),
        symbols=["AAA", "BBB"],
        as_of=date(2026, 5, 1),
        min_coverage=1.0,
    )

    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        financials = store.financial_statement_coverage(["AAA", "BBB"], as_of=date(2026, 5, 1), venue="NSE")
        valuations = store.equity_valuation_coverage(["AAA", "BBB"], as_of=date(2026, 5, 1), venue="NSE")
        shareholding = store.shareholding_coverage(["AAA", "BBB"], as_of=date(2026, 5, 1), venue="NSE")
    finally:
        store.close()

    assert report["passed"] is True
    assert report["financials"]["persisted"] == 2
    assert financials["coverage"] == 1.0
    assert valuations["coverage"] == 1.0
    assert shareholding["coverage"] == 1.0
    assert (tmp_path / "quality" / "factor_bootstrap_ingest_report.json").exists()


def test_run_security_master_ingest_persists_csv_rows(monkeypatch, tmp_path) -> None:
    csv_path = tmp_path / "securities.csv"
    csv_path.write_text(
        "\n".join(
            [
                "symbol,exchange,isin,company_name,sector,industry,listing_date,active,lot_size,tick_size,source",
                "ABC,NSE,INE000A01000,ABC Ltd,IT,Software,2020-01-01,true,1,0.05,nse",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SSE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("SSE_SQLITE_PATH", str(tmp_path / "market.db"))

    report = run_security_master_ingest(file_path=str(csv_path))

    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        rows = store.get_security_master(["ABC"])
    finally:
        store.close()

    assert report["passed"] is True
    assert report["persisted"] == 1
    assert rows[0].company_name == "ABC Ltd"
    assert rows[0].sector == "IT"


def test_run_security_master_ingest_accepts_nse_style_headers(monkeypatch, tmp_path) -> None:
    csv_path = tmp_path / "nifty.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Company Name,Industry,Symbol,Series,ISIN Code",
                "Reliance Industries Ltd.,Oil Gas & Consumable Fuels,RELIANCE,EQ,INE002A01018",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SSE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("SSE_SQLITE_PATH", str(tmp_path / "market.db"))

    report = run_security_master_ingest(file_path=str(csv_path))

    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        rows = store.get_security_master(["RELIANCE"])
    finally:
        store.close()

    assert report["passed"] is True
    assert report["persisted"] == 1
    assert rows[0].company_name == "Reliance Industries Ltd."
    assert rows[0].sector == "Oil Gas & Consumable Fuels"
    assert rows[0].isin == "INE002A01018"


def _statement(
    symbol: str,
    period_end: str,
    filing_date: str,
    revenue: float,
    net_income: float,
    source_id: str,
) -> FinancialStatementRecord:
    return FinancialStatementRecord(
        venue="NSE",
        symbol=symbol,
        period_end=date.fromisoformat(period_end),
        filing_date=date.fromisoformat(filing_date),
        statement_type="quarterly",
        revenue=revenue,
        ebit=net_income * 1.25,
        net_income=net_income,
        operating_cash_flow=net_income * 1.2,
        capex=net_income * 0.2,
        total_debt=200.0,
        equity=660.0,
        total_assets=1700.0,
        current_assets=650.0,
        current_liabilities=340.0,
        interest_expense=max(1.0, net_income * 0.12),
        source_id=source_id,
    )


def _annual_statement(
    symbol: str,
    net_income: float,
    equity: float,
) -> FinancialStatementRecord:
    return FinancialStatementRecord(
        venue="NSE",
        symbol=symbol,
        period_end=date(2026, 3, 31),
        filing_date=date(2026, 4, 20),
        statement_type="annual",
        revenue=net_income * 8.0,
        ebit=net_income * 1.2,
        net_income=net_income,
        operating_cash_flow=net_income * 1.1,
        capex=net_income * 0.2,
        total_debt=equity * 0.2,
        equity=equity,
        total_assets=equity * 2.0,
        current_assets=equity * 0.8,
        current_liabilities=equity * 0.3,
        interest_expense=max(1.0, net_income * 0.1),
        source_id="fy26",
    )
