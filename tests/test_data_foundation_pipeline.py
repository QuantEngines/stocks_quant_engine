from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Sequence

from stock_screener_engine.app import _build_data_foundation_pipeline
from stock_screener_engine.config.settings import load_settings
from stock_screener_engine.data_sources.base.interfaces import ExchangeIngestionAdapter, MarketIngestionAdapter
from stock_screener_engine.data_sources.calendar.market_calendar import MarketCalendar
from stock_screener_engine.data_sources.schemas import (
    AnnouncementRecord,
    CorporateActionRecord,
    OHLCVBar,
    ShareholdingRecord,
)
from stock_screener_engine.pipelines.data_foundation import DataFoundationPipeline
from stock_screener_engine.storage.market_data_store import MarketDataStore


def test_data_foundation_persists_canonical_store_and_quality_report(tmp_path: Path) -> None:
    settings = load_settings()
    settings = replace(
        settings,
        storage=replace(settings.storage, root_dir=str(tmp_path), sqlite_path=str(tmp_path / "market.db")),
    )
    store = MarketDataStore(settings.storage.sqlite_path)
    pipeline = DataFoundationPipeline(
        settings=settings,
        market_adapters=[_FoundationMarketAdapter()],
        exchange_adapters=[_FoundationExchangeAdapter()],
        calendar=MarketCalendar.from_dates(holidays=[date(2026, 1, 1)]),
        store=store,
    )

    report = pipeline.run(["AAA"], start=date(2026, 1, 1), end=date(2026, 1, 2))

    assert report["passed"] is True
    assert report["rows_persisted"]["security_master"] == 1
    assert report["rows_persisted"]["ohlcv_bars"] == 1
    assert report["rows_persisted"]["corporate_actions"] == 1
    assert report["rows_persisted"]["shareholding"] == 1
    assert (tmp_path / "quality" / "data_foundation_quality_report.json").exists()
    assert store.get_security_master(["AAA"])
    assert store.get_ohlcv("AAA")
    assert store.get_shareholding("AAA")
    pipeline.close()


def test_data_quality_report_reads_from_store(tmp_path: Path) -> None:
    settings = load_settings()
    settings = replace(
        settings,
        storage=replace(settings.storage, root_dir=str(tmp_path), sqlite_path=str(tmp_path / "market.db")),
    )
    store = MarketDataStore(settings.storage.sqlite_path)
    store.upsert_ohlcv([
        OHLCVBar("NSE", "AAA", "2026-01-02", 100.0, 101.0, 99.0, 100.0, 1000.0)
    ])
    pipeline = DataFoundationPipeline(settings=settings, market_adapters=[], store=store)

    report = pipeline.quality_report(["AAA"], start=date(2026, 1, 1), end=date(2026, 1, 3))

    assert report["passed"] is True
    assert report["coverage"]["coverage"] == 1.0
    assert (tmp_path / "quality" / "data_quality_report.json").exists()
    pipeline.close()


def test_data_foundation_can_return_failed_report_without_raising(tmp_path: Path) -> None:
    settings = load_settings()
    settings = replace(
        settings,
        storage=replace(settings.storage, root_dir=str(tmp_path), sqlite_path=str(tmp_path / "market.db")),
    )
    store = MarketDataStore(settings.storage.sqlite_path)
    pipeline = DataFoundationPipeline(settings=settings, market_adapters=[], store=store)

    report = pipeline.run(
        ["AAA"],
        start=date(2026, 1, 1),
        end=date(2026, 1, 2),
        raise_on_failure=False,
    )

    assert report["passed"] is False
    assert report["quality_flags"]["source_reconciliation"]["issues"][0]["symbol"] == "AAA"
    pipeline.close()


def test_app_foundation_builder_writes_provider_data_to_canonical_venue(tmp_path: Path) -> None:
    settings = load_settings()
    settings = replace(
        settings,
        storage=replace(settings.storage, root_dir=str(tmp_path), sqlite_path=str(tmp_path / "market.db")),
        runtime_data=replace(
            settings.runtime_data,
            market_provider="mock",
            market_universe=["RELIANCE"],
            canonical_venue="NSE",
        ),
    )
    pipeline = _build_data_foundation_pipeline(settings)
    try:
        pipeline.run(["RELIANCE"], start=date(2026, 1, 1), end=date(2026, 1, 2))

        assert pipeline.store.get_ohlcv("RELIANCE", venue="NSE")
        assert not pipeline.store.get_ohlcv("RELIANCE", venue="MOCK")
    finally:
        pipeline.close()


class _FoundationMarketAdapter(MarketIngestionAdapter):
    venue = "NSE"

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> list[OHLCVBar]:
        return [OHLCVBar(self.venue, symbol, "2026-01-02", 100.0, 101.0, 99.0, 100.0, 1000.0)]


class _FoundationExchangeAdapter(ExchangeIngestionAdapter):
    venue = "NSE"

    def fetch_corporate_actions(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> list[CorporateActionRecord]:
        return [
            CorporateActionRecord(
                venue=self.venue,
                symbol=symbols[0],
                action_type="split",
                ex_date="2026-01-02",
                ratio="2:1",
                source_id="1",
            )
        ]

    def fetch_shareholding(self, symbols: Sequence[str], as_of: date) -> list[ShareholdingRecord]:
        return [
            ShareholdingRecord(
                venue=self.venue,
                symbol=symbols[0],
                period_end=as_of,
                filing_date=as_of,
                promoter_pct=52.0,
                fii_pct=11.0,
                dii_pct=16.0,
                public_pct=21.0,
                source_id="q4",
            )
        ]

    def fetch_announcements(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> list[AnnouncementRecord]:
        return []
