from __future__ import annotations

from datetime import date, timedelta

from stock_screener_engine.data_sources.market.sqlite_market_data_provider import SQLiteMarketDataProvider
from stock_screener_engine.data_sources.schemas import (
    CorporateActionRecord,
    MarketSessionRecord,
    OHLCVBar,
    SecurityMasterRecord,
)
from stock_screener_engine.storage.market_data_store import MarketDataStore


def test_sqlite_provider_uses_stored_universe_and_security_master(tmp_path) -> None:
    path = tmp_path / "market.db"
    store = MarketDataStore(str(path))
    try:
        store.upsert_security_master(
            [
                SecurityMasterRecord(symbol="AAA", company_name="AAA Ltd", sector="Banks"),
                SecurityMasterRecord(symbol="BBB", company_name="BBB Ltd", sector="IT"),
            ]
        )
        store.upsert_ohlcv(
            [
                OHLCVBar("NSE", "AAA", "2026-01-02", 100.0, 104.0, 99.0, 102.0, 1_000_000.0),
                OHLCVBar("NSE", "BBB", "2026-01-02", 200.0, 205.0, 198.0, 204.0, 2_000_000.0),
            ]
        )
    finally:
        store.close()

    provider = SQLiteMarketDataProvider(
        sqlite_path=str(path),
        universe=["AAA", "BBB", "ZZZ"],
        venue="NSE",
    )
    try:
        assert provider.get_universe() == ["AAA", "BBB"]
        snapshots = provider.get_snapshots(["AAA", "BBB"])
        metadata = provider.get_company_metadata(["AAA", "BBB"])
    finally:
        provider.close()

    assert [snapshot.symbol for snapshot in snapshots] == ["AAA", "BBB"]
    assert snapshots[0].sector == "Banks"
    assert snapshots[1].sector == "IT"
    assert metadata["AAA"]["company_name"] == "AAA Ltd"
    assert metadata["BBB"]["sector"] == "IT"


def test_sqlite_provider_returns_adjusted_history_but_unadjusted_snapshot(tmp_path) -> None:
    path = tmp_path / "market.db"
    store = MarketDataStore(str(path))
    try:
        store.upsert_security_master([SecurityMasterRecord(symbol="AAA", sector="Industrials")])
        store.upsert_ohlcv(
            [
                OHLCVBar("NSE", "AAA", "2026-01-01", 100.0, 110.0, 90.0, 100.0, 1_000.0),
                OHLCVBar("NSE", "AAA", "2026-01-03", 52.0, 58.0, 51.0, 56.0, 2_000.0),
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
    finally:
        store.close()

    provider = SQLiteMarketDataProvider(sqlite_path=str(path), adjusted_history=True)
    try:
        history = provider.get_historical("AAA", "1d", date(2026, 1, 1), date(2026, 1, 3))
        snapshot = provider.get_snapshots(["AAA"])[0]
    finally:
        provider.close()

    assert history[0]["close"] == 50.0
    assert history[0]["volume"] == 2_000
    assert snapshot.close == 56.0


def test_sqlite_provider_freshness_can_warn_or_block(tmp_path) -> None:
    path = tmp_path / "market.db"
    today = date.today()
    latest_session = today - timedelta(days=1)
    stale_day = today - timedelta(days=20)
    store = MarketDataStore(str(path))
    try:
        store.upsert_security_master([SecurityMasterRecord(symbol="AAA", sector="IT")])
        store.upsert_market_sessions(
            [
                MarketSessionRecord("NSE", stale_day, True),
                MarketSessionRecord("NSE", latest_session, True),
            ]
        )
        store.upsert_ohlcv(
            [OHLCVBar("NSE", "AAA", stale_day.isoformat(), 100.0, 101.0, 99.0, 100.0, 1_000_000.0)]
        )
    finally:
        store.close()

    relaxed = SQLiteMarketDataProvider(
        sqlite_path=str(path),
        strict_freshness=False,
        max_staleness_days=3,
    )
    strict = SQLiteMarketDataProvider(
        sqlite_path=str(path),
        strict_freshness=True,
        max_staleness_days=3,
    )
    try:
        relaxed_report = relaxed.get_freshness_report(["AAA"])
        strict_report = strict.get_freshness_report(["AAA"])
    finally:
        relaxed.close()
        strict.close()

    assert relaxed_report["passed"] is True
    assert relaxed_report["warnings"]
    assert strict_report["passed"] is False
    assert strict_report["issues"]
