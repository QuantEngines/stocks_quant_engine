from __future__ import annotations

from datetime import date

from stock_screener_engine.app import run_peer_report, run_sector_peer_report
from stock_screener_engine.data_sources.financials.sqlite_financials_provider import SQLiteFinancialsProvider
from stock_screener_engine.data_sources.schemas import (
    EquityValuationRecord,
    FinancialStatementRecord,
    SecurityMasterRecord,
)
from stock_screener_engine.research.peer_comparison import (
    PeerComparisonBuilder,
    render_peer_markdown,
    render_sector_peer_markdown,
)
from stock_screener_engine.storage.market_data_store import MarketDataStore


def test_peer_comparison_ranks_value_quality_growth_and_composite(tmp_path) -> None:
    path = tmp_path / "market.db"
    _seed_peer_store(path)

    store = MarketDataStore(str(path))
    financials = SQLiteFinancialsProvider(sqlite_path=str(path), store=store)
    try:
        report = PeerComparisonBuilder(store, financials).build("AAA", as_of=date(2026, 5, 1))
    finally:
        financials.close()

    assert report.peer_count == 3
    assert report.target is not None
    assert report.target.growth_rank == 1
    assert "BBB" in report.valuation_leaders
    assert report.composite_leaders[0] == "AAA"
    assert "Peer Comparison" in render_peer_markdown(report)


def test_sector_peer_comparison_renders_leaders(tmp_path) -> None:
    path = tmp_path / "market.db"
    _seed_peer_store(path)

    store = MarketDataStore(str(path))
    financials = SQLiteFinancialsProvider(sqlite_path=str(path), store=store)
    try:
        report = PeerComparisonBuilder(store, financials).build_sector("it", as_of=date(2026, 5, 1))
    finally:
        financials.close()

    assert report.peer_count == 3
    assert report.growth_leaders[0] == "AAA"
    assert "# it Peer Rankings" in render_sector_peer_markdown(report)


def test_run_peer_report_uses_canonical_store(monkeypatch, tmp_path) -> None:
    path = tmp_path / "market.db"
    _seed_peer_store(path)
    monkeypatch.setenv("SSE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("SSE_SQLITE_PATH", str(path))

    payload = run_peer_report("AAA", as_of=date(2026, 5, 1))
    sector_payload = run_sector_peer_report("IT", as_of=date(2026, 5, 1), output_format="markdown")

    assert payload["symbol"] == "AAA"
    assert payload["target"]["growth_rank"] == 1
    assert "markdown" in sector_payload


def _seed_peer_store(path) -> None:
    store = MarketDataStore(str(path))
    try:
        store.upsert_security_master(
            [
                SecurityMasterRecord(symbol="AAA", company_name="AAA Ltd", sector="IT", industry="Software"),
                SecurityMasterRecord(symbol="BBB", company_name="BBB Ltd", sector="IT", industry="Services"),
                SecurityMasterRecord(symbol="CCC", company_name="CCC Ltd", sector="IT", industry="Hardware"),
            ]
        )
        store.upsert_financial_statements(
            [
                _statement("AAA", "2025-03-31", "2025-04-20", revenue=1000.0, net_income=100.0, equity=700.0, source_id="aaa25"),
                _statement("AAA", "2026-03-31", "2026-04-20", revenue=1300.0, net_income=150.0, equity=750.0, source_id="aaa26"),
                _statement("BBB", "2025-03-31", "2025-04-20", revenue=1000.0, net_income=90.0, equity=480.0, source_id="bbb25"),
                _statement("BBB", "2026-03-31", "2026-04-20", revenue=1100.0, net_income=100.0, equity=500.0, source_id="bbb26"),
                _statement("CCC", "2025-03-31", "2025-04-20", revenue=1000.0, net_income=100.0, equity=420.0, source_id="ccc25"),
                _statement("CCC", "2026-03-31", "2026-04-20", revenue=900.0, net_income=80.0, equity=400.0, source_id="ccc26"),
            ]
        )
        store.upsert_equity_valuations(
            [
                EquityValuationRecord("NSE", "AAA", date(2026, 4, 30), market_cap=3000.0),
                EquityValuationRecord("NSE", "BBB", date(2026, 4, 30), market_cap=1000.0),
                EquityValuationRecord("NSE", "CCC", date(2026, 4, 30), market_cap=1600.0),
            ]
        )
    finally:
        store.close()


def _statement(
    symbol: str,
    period_end: str,
    filing_date: str,
    revenue: float,
    net_income: float,
    equity: float,
    source_id: str,
) -> FinancialStatementRecord:
    return FinancialStatementRecord(
        venue="NSE",
        symbol=symbol,
        period_end=date.fromisoformat(period_end),
        filing_date=date.fromisoformat(filing_date),
        statement_type="annual",
        revenue=revenue,
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
        source_id=source_id,
    )
