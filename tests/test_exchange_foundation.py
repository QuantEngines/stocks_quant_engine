from __future__ import annotations

from datetime import date

from stock_screener_engine.data_sources.exchange.delivery_csv_loader import load_delivery_turnover_csv
from stock_screener_engine.data_sources.market.sqlite_market_data_provider import SQLiteMarketDataProvider
from stock_screener_engine.data_sources.schemas import DeliveryTurnoverRecord, OHLCVBar, SecurityMasterRecord
from stock_screener_engine.pipelines.exchange_foundation import build_exchange_foundation_status
from stock_screener_engine.storage.market_data_store import MarketDataStore


def test_delivery_csv_loader_accepts_nse_style_headers(tmp_path) -> None:
    path = tmp_path / "delivery.csv"
    path.write_text(
        "SYMBOL,DATE1,TTL_TRD_QNTY,DELIV_QTY,DELIV_PER\n"
        "RELIANCE,2026-05-28,100000,65000,65.0\n",
        encoding="utf-8",
    )

    rows = load_delivery_turnover_csv(path, venue="NSE")

    assert rows == [
        DeliveryTurnoverRecord(
            venue="NSE",
            symbol="RELIANCE",
            trade_date=date(2026, 5, 28),
            traded_quantity=100000.0,
            delivery_quantity=65000.0,
            delivery_pct=65.0,
            source_id="delivery.csv",
        )
    ]


def test_delivery_turnover_updates_canonical_snapshot(tmp_path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    store.upsert_security_master([
        SecurityMasterRecord(symbol="RELIANCE", exchange="NSE", sector="Energy", company_name="Reliance Industries")
    ])
    store.upsert_ohlcv([
        OHLCVBar(
            venue="NSE",
            symbol="RELIANCE",
            ts="2026-05-28",
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=100000.0,
        )
    ])
    store.upsert_delivery_turnover([
        DeliveryTurnoverRecord(
            venue="NSE",
            symbol="RELIANCE",
            trade_date=date(2026, 5, 28),
            traded_quantity=100000.0,
            delivery_quantity=65000.0,
            delivery_pct=65.0,
        )
    ])
    provider = SQLiteMarketDataProvider(sqlite_path=str(tmp_path / "market.db"), universe=["RELIANCE"], store=store)

    snapshot = provider.get_snapshots(["RELIANCE"])[0]

    assert snapshot.delivery_ratio == 0.65


def test_exchange_foundation_status_reports_delivery_coverage(tmp_path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    store.upsert_security_master([SecurityMasterRecord(symbol="ITC", exchange="NSE", sector="FMCG")])
    store.upsert_delivery_turnover([
        DeliveryTurnoverRecord(
            venue="NSE",
            symbol="ITC",
            trade_date=date(2026, 5, 28),
            traded_quantity=1000.0,
            delivery_quantity=500.0,
            delivery_pct=50.0,
        )
    ])

    report = build_exchange_foundation_status(
        store=store,
        symbols=["ITC", "RELIANCE"],
        as_of=date(2026, 5, 28),
        start=date(2026, 1, 1),
        venue="NSE",
    )
    rows = {row["domain"]: row for row in report["domains"]}

    assert rows["delivery_turnover"]["coverage"] == 0.5
    assert rows["delivery_turnover"]["status"] == "thin"
    assert "# NSE/BSE Exchange Foundation Status" in report["markdown"]
