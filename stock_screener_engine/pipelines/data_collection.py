"""Canonical data collection pipeline for exchange-sourced Indian market data."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from typing import Any, Sequence

from stock_screener_engine.config.settings import AppSettings
from stock_screener_engine.data_sources.base.interfaces import ExchangeIngestionAdapter, MarketIngestionAdapter
from stock_screener_engine.data_sources.schemas import (
    AnnouncementRecord,
    CorporateActionRecord,
    OHLCVBar,
    ShareholdingRecord,
)
from stock_screener_engine.monitoring.data_quality import DataQualityChecker
from stock_screener_engine.storage.local_files import LocalFileStorage


class DataCollectionPipeline:
    """Collect normalized data from exchange adapters and persist auditable files.

    The pipeline stores canonical CSVs in ``data/cleaned`` and a quality report
    in ``data/quality``. Individual source failures are captured in the report so
    another venue or broker-backed source can still provide coverage.
    """

    def __init__(
        self,
        settings: AppSettings,
        market_adapters: Sequence[MarketIngestionAdapter],
        exchange_adapters: Sequence[ExchangeIngestionAdapter],
        quality_checker: DataQualityChecker | None = None,
    ) -> None:
        self.file_store = LocalFileStorage(settings.storage.root_dir)
        self.market_adapters = list(market_adapters)
        self.exchange_adapters = list(exchange_adapters)
        self.quality = quality_checker or DataQualityChecker()

    def run(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        interval: str = "1d",
    ) -> dict[str, object]:
        run_at = datetime.utcnow().isoformat() + "Z"
        symbols = [s.strip().upper() for s in symbols if s.strip()]
        ohlcv_bars, market_errors = self._collect_ohlcv(symbols, start, end, interval)
        corporate_actions, shareholding, announcements, exchange_errors = self._collect_exchange_events(
            symbols,
            start,
            end,
        )

        output_files = {
            "ohlcv": str(
                self.file_store.save_rows_csv(
                    _record_rows(ohlcv_bars),
                    filename=f"ohlcv_{start.isoformat()}_{end.isoformat()}.csv",
                    subdir="cleaned",
                )
            ),
            "corporate_actions": str(
                self.file_store.save_rows_csv(
                    _record_rows(corporate_actions),
                    filename=f"corporate_actions_{start.isoformat()}_{end.isoformat()}.csv",
                    subdir="cleaned",
                )
            ),
            "shareholding": str(
                self.file_store.save_rows_csv(
                    _record_rows(shareholding),
                    filename=f"shareholding_{end.isoformat()}.csv",
                    subdir="cleaned",
                )
            ),
            "announcements": str(
                self.file_store.save_rows_csv(
                    _record_rows(announcements),
                    filename=f"announcements_{start.isoformat()}_{end.isoformat()}.csv",
                    subdir="cleaned",
                )
            ),
        }

        ohlcv_quality = self.quality.validate_ohlcv_bars(ohlcv_bars, requested_symbols=symbols)
        report = {
            "pipeline": "data_collection",
            "run_at": run_at,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "interval": interval,
            "symbols_requested": len(symbols),
            "passed": ohlcv_quality.passed,
            "quality_flags": {
                "ohlcv": ohlcv_quality.to_dict(),
            },
            "source_errors": [*market_errors, *exchange_errors],
            "row_counts": {
                "ohlcv": len(ohlcv_bars),
                "corporate_actions": len(corporate_actions),
                "shareholding": len(shareholding),
                "announcements": len(announcements),
            },
            "output_files": output_files,
        }
        self.file_store.save_json(report, filename="data_collection_quality_report.json", subdir="quality")
        if not ohlcv_quality.passed:
            raise RuntimeError("Data collection blocked by OHLCV quality issues")
        return report

    def _collect_ohlcv(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        interval: str,
    ) -> tuple[list[OHLCVBar], list[str]]:
        bars: list[OHLCVBar] = []
        errors: list[str] = []
        for adapter in self.market_adapters:
            venue = _adapter_name(adapter)
            for symbol in symbols:
                try:
                    bars.extend(adapter.fetch_ohlcv(symbol, start=start, end=end, interval=interval))
                except Exception as exc:  # pragma: no cover - source-specific network failures
                    errors.append(f"{venue}:{symbol}: {exc}")
        return bars, errors

    def _collect_exchange_events(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> tuple[
        list[CorporateActionRecord],
        list[ShareholdingRecord],
        list[AnnouncementRecord],
        list[str],
    ]:
        corporate_actions: list[CorporateActionRecord] = []
        shareholding: list[ShareholdingRecord] = []
        announcements: list[AnnouncementRecord] = []
        errors: list[str] = []
        for adapter in self.exchange_adapters:
            venue = _adapter_name(adapter)
            try:
                corporate_actions.extend(adapter.fetch_corporate_actions(symbols, start=start, end=end))
            except Exception as exc:  # pragma: no cover - source-specific network failures
                errors.append(f"{venue}:corporate_actions: {exc}")
            try:
                shareholding.extend(adapter.fetch_shareholding(symbols, as_of=end))
            except Exception as exc:  # pragma: no cover - source-specific network failures
                errors.append(f"{venue}:shareholding: {exc}")
            try:
                announcements.extend(adapter.fetch_announcements(symbols, start=start, end=end))
            except Exception as exc:  # pragma: no cover - source-specific network failures
                errors.append(f"{venue}:announcements: {exc}")
        return corporate_actions, shareholding, announcements, errors


def _record_rows(records: Sequence[Any]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        row = asdict(record)
        for key, value in list(row.items()):
            if isinstance(value, date):
                row[key] = value.isoformat()
        rows.append(row)
    return rows


def _adapter_name(adapter: object) -> str:
    return str(getattr(adapter, "venue", adapter.__class__.__name__))
