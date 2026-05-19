from __future__ import annotations

from datetime import date

from stock_screener_engine.data_sources.calendar.market_calendar import MarketCalendar
from stock_screener_engine.data_sources.schemas import (
    CorporateActionRecord,
    EquityValuationRecord,
    FinancialStatementRecord,
    OHLCVBar,
    SecurityMasterRecord,
    ShareholdingRecord,
)
from stock_screener_engine.storage.market_data_store import MarketDataStore


def test_market_data_store_persists_security_calendar_and_ohlcv(tmp_path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        store.upsert_security_master([
            SecurityMasterRecord(symbol="AAA", isin="INE000A01000", company_name="AAA Ltd")
        ])
        sessions = MarketCalendar(holidays=frozenset({date(2026, 1, 1)})).sessions(
            date(2026, 1, 1),
            date(2026, 1, 5),
        )
        store.upsert_market_sessions(sessions)
        store.upsert_ohlcv([
            OHLCVBar("NSE", "AAA", "2026-01-02", 100.0, 105.0, 99.0, 102.0, 1000.0)
        ])

        securities = store.get_security_master(["AAA"])
        stored_sessions = store.get_market_sessions("NSE", date(2026, 1, 1), date(2026, 1, 5))
        bars = store.get_ohlcv("AAA", start=date(2026, 1, 1), end=date(2026, 1, 5))

        assert securities[0].isin == "INE000A01000"
        assert stored_sessions[0].is_trading_day is False
        assert bars[0].close == 102.0
    finally:
        store.close()


def test_market_data_store_lists_active_sector_peers_and_metadata(tmp_path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        store.upsert_security_master(
            [
                SecurityMasterRecord(symbol="AAA", company_name="AAA Ltd", sector="IT", industry="Software"),
                SecurityMasterRecord(symbol="BBB", company_name="BBB Ltd", sector="IT", industry="Services"),
                SecurityMasterRecord(symbol="CCC", company_name="CCC Ltd", sector="Banks", active=False),
            ]
        )

        active_it = store.list_active_securities(sectors=["it"])
        metadata = store.company_metadata(["AAA", "CCC"])

        assert [record.symbol for record in active_it] == ["AAA", "BBB"]
        assert metadata["AAA"]["company_name"] == "AAA Ltd"
        assert metadata["AAA"]["industry"] == "Software"
        assert metadata["CCC"]["active"] is False
    finally:
        store.close()


def test_store_returns_split_adjusted_history(tmp_path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        store.upsert_ohlcv(
            [
                OHLCVBar("NSE", "AAA", "2026-01-01", 100.0, 110.0, 90.0, 100.0, 1000.0),
                OHLCVBar("NSE", "AAA", "2026-01-03", 50.0, 55.0, 45.0, 50.0, 2000.0),
            ]
        )
        store.upsert_corporate_actions(
            [
                CorporateActionRecord(
                    venue="NSE",
                    symbol="AAA",
                    action_type="split",
                    ex_date="2026-01-02",
                    ratio="2:1",
                    source_id="split-1",
                )
            ]
        )

        adjusted = store.get_ohlcv("AAA", adjusted=True)

        assert adjusted[0].close == 50.0
        assert adjusted[0].volume == 2000.0
        assert adjusted[1].close == 50.0
    finally:
        store.close()


def test_market_data_store_persists_point_in_time_financial_statements(tmp_path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        store.upsert_financial_statements(
            [
                FinancialStatementRecord(
                    venue="NSE",
                    symbol="AAA",
                    period_end=date(2025, 3, 31),
                    filing_date=date(2025, 4, 20),
                    statement_type="annual",
                    revenue=1000.0,
                    ebit=180.0,
                    net_income=120.0,
                    operating_cash_flow=150.0,
                    capex=40.0,
                    total_debt=200.0,
                    equity=600.0,
                    total_assets=1400.0,
                    current_assets=500.0,
                    current_liabilities=300.0,
                    interest_expense=30.0,
                    source_id="fy25",
                ),
                FinancialStatementRecord(
                    venue="NSE",
                    symbol="AAA",
                    period_end=date(2026, 3, 31),
                    filing_date=date(2026, 4, 20),
                    statement_type="annual",
                    revenue=1200.0,
                    ebit=240.0,
                    net_income=180.0,
                    operating_cash_flow=230.0,
                    capex=50.0,
                    total_debt=180.0,
                    equity=720.0,
                    total_assets=1600.0,
                    current_assets=600.0,
                    current_liabilities=320.0,
                    interest_expense=25.0,
                    source_id="fy26",
                ),
            ]
        )

        older = store.latest_financial_statement_as_of("AAA", as_of=date(2026, 1, 1), venue="NSE")
        latest = store.latest_financial_statement_as_of("AAA", as_of=date(2026, 5, 1), venue="NSE")
        coverage = store.financial_statement_coverage(["AAA", "BBB"], as_of=date(2026, 5, 1), venue="NSE")

        assert older is not None
        assert older.period_end == date(2025, 3, 31)
        assert latest is not None
        assert latest.period_end == date(2026, 3, 31)
        assert coverage["coverage"] == 0.5
    finally:
        store.close()


def test_market_data_store_persists_point_in_time_equity_valuations(tmp_path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        store.upsert_equity_valuations(
            [
                EquityValuationRecord(
                    venue="NSE",
                    symbol="AAA",
                    as_of=date(2026, 4, 30),
                    market_cap=9000.0,
                    shares_outstanding=100.0,
                    source_id="apr",
                ),
                EquityValuationRecord(
                    venue="NSE",
                    symbol="AAA",
                    as_of=date(2026, 5, 31),
                    market_cap=10000.0,
                    shares_outstanding=100.0,
                    source_id="may",
                ),
            ]
        )

        older = store.latest_equity_valuation_as_of("AAA", as_of=date(2026, 5, 1), venue="NSE")
        latest = store.latest_equity_valuation_as_of("AAA", as_of=date(2026, 6, 1), venue="NSE")
        coverage = store.equity_valuation_coverage(["AAA", "BBB"], as_of=date(2026, 6, 1), venue="NSE")

        assert older is not None
        assert older.market_cap == 9000.0
        assert latest is not None
        assert latest.market_cap == 10000.0
        assert coverage["coverage"] == 0.5
    finally:
        store.close()


def test_market_data_store_persists_point_in_time_shareholding(tmp_path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    try:
        store.upsert_shareholding(
            [
                ShareholdingRecord(
                    venue="NSE",
                    symbol="AAA",
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
                    symbol="AAA",
                    period_end=date(2026, 3, 31),
                    filing_date=date(2026, 4, 20),
                    promoter_pct=52.0,
                    fii_pct=11.0,
                    dii_pct=16.0,
                    public_pct=21.0,
                    source_id="q4",
                ),
            ]
        )

        older = store.latest_shareholding_as_of("AAA", as_of=date(2026, 2, 1), venue="NSE")
        latest = store.latest_shareholding_as_of("AAA", as_of=date(2026, 5, 1), venue="NSE")
        coverage = store.shareholding_coverage(["AAA", "BBB"], as_of=date(2026, 5, 1), venue="NSE")

        assert older is not None
        assert older.promoter_pct == 50.0
        assert latest is not None
        assert latest.promoter_pct == 52.0
        assert coverage["coverage"] == 0.5
    finally:
        store.close()
