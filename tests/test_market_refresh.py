from __future__ import annotations

from dataclasses import replace
from datetime import date

from stock_screener_engine.app import _normalize_refresh_daily_bars, _symbols_needing_retry
from stock_screener_engine.config.settings import load_settings
from stock_screener_engine.data_sources.schemas import OHLCVBar
from stock_screener_engine.storage.market_data_store import MarketDataStore


def test_symbols_needing_retry_uses_source_errors_and_quality_issues() -> None:
    report = {
        "source_errors": ["NSE:AAA: Too many requests"],
        "coverage": {"missing_symbols": ["BBB"], "rows_by_symbol": {}},
        "quality_flags": {
            "ohlcv": {"warnings": ["Missing OHLCV bars for: CCC"]},
            "source_reconciliation": {
                "issues": [
                    {
                        "symbol": "DDD",
                        "severity": "error",
                        "message": "No bars observed from any source",
                    },
                    {
                        "symbol": "EEE",
                        "severity": "warning",
                        "message": "Volume divergence",
                    },
                ]
            },
        },
    }

    retry = _symbols_needing_retry(report, ["AAA", "BBB", "CCC", "DDD", "EEE"])

    assert retry == {"AAA", "BBB", "CCC", "DDD"}


def test_normalize_refresh_daily_bars_repairs_requested_universe(tmp_path) -> None:
    settings = load_settings()
    settings = replace(
        settings,
        storage=replace(settings.storage, root_dir=str(tmp_path), sqlite_path=str(tmp_path / "market.db")),
    )
    store = MarketDataStore(settings.storage.sqlite_path)
    try:
        store.upsert_ohlcv([OHLCVBar("NSE", "AAA", "2026-05-18", 100.0, 101.0, 99.0, 100.0, 1000.0)])
        store.conn.execute(
            """
            INSERT INTO ohlcv_bars(venue, symbol, ts, interval, open, high, low, close, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "NSE",
                "AAA",
                "2026-05-18T00:00:00+05:30",
                "1d",
                100.0,
                101.0,
                99.0,
                100.0,
                1000.0,
                "legacy_overlap",
            ),
        )
        store.conn.commit()
    finally:
        store.close()

    report = _normalize_refresh_daily_bars(settings, ["AAA"], "1d")

    store = MarketDataStore(settings.storage.sqlite_path)
    try:
        bars = store.get_ohlcv("AAA", start=date(2026, 5, 18), end=date(2026, 5, 18))
    finally:
        store.close()

    assert report["rows_deleted"] == 1
    assert len(bars) == 1
