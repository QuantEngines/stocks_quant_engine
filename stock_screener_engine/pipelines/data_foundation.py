"""Data foundation pipeline: canonical storage, calendar, and reconciliation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from typing import Any, Sequence

from stock_screener_engine.config.settings import AppSettings
from stock_screener_engine.data_sources.base.interfaces import ExchangeIngestionAdapter, MarketIngestionAdapter
from stock_screener_engine.data_sources.calendar.market_calendar import MarketCalendar
from stock_screener_engine.data_sources.schemas import (
    CorporateActionRecord,
    OHLCVBar,
    SecurityMasterRecord,
    ShareholdingRecord,
)
from stock_screener_engine.data_sources.security_master.provider import build_minimal_security_master
from stock_screener_engine.monitoring.data_quality import DataQualityChecker
from stock_screener_engine.monitoring.source_reconciliation import SourceReconciler
from stock_screener_engine.storage.local_files import LocalFileStorage
from stock_screener_engine.storage.market_data_store import MarketDataStore


class DataFoundationPipeline:
    """Create and validate the canonical data layer used by research pipelines."""

    def __init__(
        self,
        settings: AppSettings,
        market_adapters: Sequence[MarketIngestionAdapter],
        exchange_adapters: Sequence[ExchangeIngestionAdapter] = (),
        security_master_records: Sequence[SecurityMasterRecord] | None = None,
        calendar: MarketCalendar | None = None,
        store: MarketDataStore | None = None,
        quality_checker: DataQualityChecker | None = None,
        reconciler: SourceReconciler | None = None,
    ) -> None:
        self.settings = settings
        self.market_adapters = list(market_adapters)
        self.exchange_adapters = list(exchange_adapters)
        self.security_master_records = list(security_master_records or [])
        self.calendar = calendar or MarketCalendar(venue="NSE")
        self.store = store or MarketDataStore(settings.storage.sqlite_path)
        self.file_store = LocalFileStorage(settings.storage.root_dir)
        self.quality = quality_checker or DataQualityChecker()
        self.reconciler = reconciler or SourceReconciler()

    def run(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        interval: str = "1d",
        raise_on_failure: bool = True,
    ) -> dict[str, object]:
        symbols = [s.strip().upper() for s in symbols if s.strip()]
        run_at = datetime.utcnow().isoformat() + "Z"
        security_records = self.security_master_records or build_minimal_security_master(symbols)
        sessions = self.calendar.sessions(start, end)
        bars, market_errors = self._collect_bars(symbols, start, end, interval)
        corporate_actions, action_errors = self._collect_corporate_actions(symbols, start, end)
        shareholding, shareholding_errors = self._collect_shareholding(symbols, as_of=end)

        security_count = self.store.upsert_security_master(security_records)
        session_count = self.store.upsert_market_sessions(sessions)
        bar_count = self.store.upsert_ohlcv(bars, interval=interval, source="data_foundation")
        action_count = self.store.upsert_corporate_actions(corporate_actions)
        shareholding_count = self.store.upsert_shareholding(shareholding)

        quality_report = self.quality.validate_ohlcv_bars(bars, requested_symbols=symbols)
        reconciliation_report = self.reconciler.reconcile(bars, requested_symbols=symbols)
        coverage = self.store.coverage_summary(symbols=symbols, start=start, end=end, interval=interval)
        report = {
            "pipeline": "data_foundation",
            "run_at": run_at,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "interval": interval,
            "passed": quality_report.passed and reconciliation_report.passed,
            "symbols_requested": len(symbols),
            "rows_persisted": {
                "security_master": security_count,
                "market_sessions": session_count,
                "ohlcv_bars": bar_count,
                "corporate_actions": action_count,
                "shareholding": shareholding_count,
            },
            "quality_flags": {
                "ohlcv": quality_report.to_dict(),
                "source_reconciliation": reconciliation_report.to_dict(),
            },
            "coverage": coverage,
            "source_errors": [*market_errors, *action_errors, *shareholding_errors],
        }
        self.file_store.save_json(report, filename="data_foundation_quality_report.json", subdir="quality")
        if not report["passed"] and raise_on_failure:
            raise RuntimeError("Data foundation blocked by quality or reconciliation issues")
        return report

    def quality_report(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        interval: str = "1d",
    ) -> dict[str, object]:
        symbols = [s.strip().upper() for s in symbols if s.strip()]
        bars: list[OHLCVBar] = []
        for symbol in symbols:
            bars.extend(self.store.get_ohlcv(symbol=symbol, start=start, end=end, interval=interval))
        quality_report = self.quality.validate_ohlcv_bars(bars, requested_symbols=symbols)
        reconciliation_report = self.reconciler.reconcile(bars, requested_symbols=symbols)
        coverage = self.store.coverage_summary(symbols=symbols, start=start, end=end, interval=interval)
        report = {
            "pipeline": "data_quality",
            "run_at": datetime.utcnow().isoformat() + "Z",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "interval": interval,
            "passed": quality_report.passed and reconciliation_report.passed,
            "quality_flags": {
                "ohlcv": quality_report.to_dict(),
                "source_reconciliation": reconciliation_report.to_dict(),
            },
            "coverage": coverage,
        }
        self.file_store.save_json(report, filename="data_quality_report.json", subdir="quality")
        return report

    def _collect_bars(
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
                    bars.extend(adapter.fetch_ohlcv(symbol=symbol, start=start, end=end, interval=interval))
                except Exception as exc:  # pragma: no cover - live-source dependent
                    errors.append(f"{venue}:{symbol}: {exc}")
        return bars, errors

    def _collect_corporate_actions(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> tuple[list[CorporateActionRecord], list[str]]:
        actions: list[CorporateActionRecord] = []
        errors: list[str] = []
        for adapter in self.exchange_adapters:
            venue = _adapter_name(adapter)
            try:
                actions.extend(adapter.fetch_corporate_actions(symbols, start=start, end=end))
            except Exception as exc:  # pragma: no cover - live-source dependent
                errors.append(f"{venue}:corporate_actions: {exc}")
        return actions, errors

    def _collect_shareholding(
        self,
        symbols: Sequence[str],
        as_of: date,
    ) -> tuple[list[ShareholdingRecord], list[str]]:
        records: list[ShareholdingRecord] = []
        errors: list[str] = []
        for adapter in self.exchange_adapters:
            venue = _adapter_name(adapter)
            try:
                records.extend(adapter.fetch_shareholding(symbols, as_of=as_of))
            except Exception as exc:  # pragma: no cover - live-source dependent
                errors.append(f"{venue}:shareholding: {exc}")
        return records, errors

    def close(self) -> None:
        self.store.close()


def serialise_records(records: Sequence[Any]) -> list[dict]:
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
