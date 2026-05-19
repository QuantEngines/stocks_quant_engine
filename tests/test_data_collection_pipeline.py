from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Sequence

from stock_screener_engine.config.settings import load_settings
from stock_screener_engine.data_sources.base.interfaces import ExchangeIngestionAdapter, MarketIngestionAdapter
from stock_screener_engine.data_sources.schemas import (
    AnnouncementRecord,
    CorporateActionRecord,
    OHLCVBar,
    ShareholdingRecord,
)
from stock_screener_engine.pipelines.data_collection import DataCollectionPipeline


def test_data_collection_pipeline_persists_outputs_and_quality(tmp_path: Path) -> None:
    settings = load_settings()
    settings = replace(
        settings,
        storage=replace(
            settings.storage,
            root_dir=str(tmp_path),
            sqlite_path=str(tmp_path / "metadata.db"),
        ),
    )
    start = date(2026, 1, 1)
    end = date(2026, 1, 2)

    report = DataCollectionPipeline(
        settings=settings,
        market_adapters=[_FakeMarketAdapter()],
        exchange_adapters=[_FakeExchangeAdapter()],
    ).run(["AAA"], start=start, end=end)

    assert report["passed"] is True
    assert report["row_counts"] == {
        "ohlcv": 1,
        "corporate_actions": 1,
        "shareholding": 1,
        "announcements": 1,
    }
    assert (tmp_path / "cleaned" / "ohlcv_2026-01-01_2026-01-02.csv").exists()
    assert (tmp_path / "quality" / "data_collection_quality_report.json").exists()


class _FakeMarketAdapter(MarketIngestionAdapter):
    venue = "TEST"

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> list[OHLCVBar]:
        return [
            OHLCVBar(
                venue=self.venue,
                symbol=symbol,
                ts=start.isoformat(),
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=1_000_000.0,
            )
        ]


class _FakeExchangeAdapter(ExchangeIngestionAdapter):
    venue = "TEST"

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
                action_type="dividend",
                ex_date=end.isoformat(),
                cash_amount=1.0,
            )
        ]

    def fetch_shareholding(self, symbols: Sequence[str], as_of: date) -> list[ShareholdingRecord]:
        return [
            ShareholdingRecord(
                venue=self.venue,
                symbol=symbols[0],
                period_end=as_of,
                filing_date=as_of,
                promoter_pct=50.0,
                fii_pct=10.0,
                dii_pct=15.0,
                public_pct=25.0,
            )
        ]

    def fetch_announcements(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> list[AnnouncementRecord]:
        return [
            AnnouncementRecord(
                venue=self.venue,
                symbol=symbols[0],
                published_at=end.isoformat(),
                category="result",
                subject="Quarterly results",
                url="https://example.test/result.pdf",
                source_id="1",
            )
        ]
