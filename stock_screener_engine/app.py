"""Application entry helpers for pipeline execution."""

from __future__ import annotations

import csv
import logging
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import date, datetime
from pathlib import Path
from typing import cast

from stock_screener_engine.config.settings import AppSettings, load_settings
from stock_screener_engine.config.startup_validation import validate_startup_settings
from stock_screener_engine.backtest.costs import IndianEquityCostModel
from stock_screener_engine.core.entities import FeatureVector, ScoreCard, SignalResult
from stock_screener_engine.data_sources.broker.factory import build_broker_adapters
from stock_screener_engine.data_sources.filings.exchange_filings_provider import ExchangeFilingsProvider
from stock_screener_engine.data_sources.filings.null_filings_provider import NullFilingsProvider
from stock_screener_engine.data_sources.filings.filings_adapter import FilingsAdapter
from stock_screener_engine.data_sources.news.generic_news_adapter import GenericNewsAdapter
from stock_screener_engine.data_sources.news.free_news_provider import FreeRSSNewsProvider
from stock_screener_engine.data_sources.transcripts.null_transcripts import NullTranscriptProvider
from stock_screener_engine.data_sources.transcripts.transcripts_adapter import TranscriptsAdapter
from stock_screener_engine.llm.base.factory import build_llm_client
from stock_screener_engine.llm.extraction.document_classifier import LLMDocumentClassifier
from stock_screener_engine.llm.extraction.event_extractor import LLMEventExtractor
from stock_screener_engine.llm.extraction.management_tone_extractor import LLMManagementToneExtractor
from stock_screener_engine.llm.extraction.sentiment_extractor import LLMSentimentExtractor
from stock_screener_engine.nlp.event_engine.aggregation import EventFeatureAggregator
from stock_screener_engine.nlp.event_engine.audit import LowConfidenceAuditSink
from stock_screener_engine.nlp.event_engine.pipeline import TextIntelligencePipeline
from stock_screener_engine.nlp.ingestion.document_ingestor import TextDocumentIngestor
from stock_screener_engine.nlp.ingestion.health_reporting import IngestionHealthSink
from stock_screener_engine.pipelines.backtest_readiness import (
    BacktestReadinessPipeline,
    BacktestReadinessThresholds,
)
from stock_screener_engine.pipelines.backtest_dataset import BacktestDatasetPipeline
from stock_screener_engine.pipelines.daily_batch import DailyBatchPipeline
from stock_screener_engine.pipelines.data_foundation import DataFoundationPipeline
from stock_screener_engine.pipelines.document_pipeline import DocumentIntelligencePipeline
from stock_screener_engine.pipelines.factor_bootstrap import FactorBootstrapPipeline
from stock_screener_engine.pipelines.intraday_update import IntradayUpdatePipeline
from stock_screener_engine.pipelines.live_invalidation_daily import run_live_invalidation_daily_job
from stock_screener_engine.reporting.signal_report import (
    build_signal_reports,
    render_signal_markdown,
    signal_reports_to_console_rows,
)
from stock_screener_engine.research.company_deepdive.report import CompanyDeepDiveBuilder
from stock_screener_engine.research.peer_comparison import (
    PeerComparisonBuilder,
    render_peer_markdown,
    render_sector_peer_markdown,
)
from stock_screener_engine.sector.sector_report import SectorIntelligenceBuilder
from stock_screener_engine.storage.local_files import LocalFileStorage
from stock_screener_engine.storage.market_data_store import MarketDataStore


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_daily(settings: AppSettings) -> dict[str, object]:
    validate_startup_settings(settings)
    market = _build_market_provider(settings)
    financials = _build_financials_provider(settings)
    text = _build_text_provider(settings)
    text_pipeline = _build_text_pipeline(settings, text)
    pipeline = DailyBatchPipeline(
        settings=settings,
        market_data=market,
        text_data=text,
        financials=financials,
        text_pipeline=text_pipeline,
    )
    try:
        return pipeline.run()
    finally:
        pipeline.close()


def run_intraday(settings: AppSettings) -> dict[str, object]:
    validate_startup_settings(settings)
    market = _build_market_provider(settings)
    financials = _build_financials_provider(settings)
    text = _build_text_provider(settings)
    text_pipeline = _build_text_pipeline(settings, text)
    pipeline = IntradayUpdatePipeline(
        settings=settings,
        market_data=market,
        text_data=text,
        financials=financials,
        text_pipeline=text_pipeline,
    )
    return pipeline.run()


def run_live_invalidation_daily(settings: AppSettings) -> dict[str, object]:
    """Run live invalidation on currently open broker positions.

    Reports are persisted as date-stamped JSON and CSV under `data/signals`.
    """
    validate_startup_settings(settings)
    configure_logging(settings.log_level)

    return run_live_invalidation_daily_job(settings)


def summarize_brokers(settings: AppSettings) -> dict[str, bool]:
    adapters = build_broker_adapters(settings)
    return {name: adapter.is_enabled() for name, adapter in adapters.items()}


def run_broker_health(
    start: date,
    end: date,
    symbols: list[str] | None = None,
    universe_file: str | None = None,
    config_path: str | None = None,
    sources: list[str] | None = None,
    interval: str = "1d",
    sample_size: int | None = None,
    price_tolerance_pct: float = 1.0,
) -> dict[str, object]:
    """Compare broker market-data health without placing orders."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)
    resolved_symbols, _ = _resolve_runtime_universe(
        settings=settings,
        symbols=symbols,
        universe_file=universe_file,
    )
    if sample_size is not None and sample_size > 0:
        resolved_symbols = resolved_symbols[:sample_size]

    broker_sources = _resolve_broker_health_sources(sources)
    adapters = build_broker_adapters(settings)
    source_reports: dict[str, dict[str, object]] = {}
    symbol_reports = {
        symbol: {
            "symbol": symbol,
            "sources": {},
            "quote_mismatch_pct": 0.0,
            "historical_close_mismatch_pct": 0.0,
            "preferred_source": "",
        }
        for symbol in resolved_symbols
    }

    for source_name, adapter_key in broker_sources:
        adapter = adapters.get(adapter_key)
        source_report = _new_broker_source_report(source_name, len(resolved_symbols))
        source_reports[source_name] = source_report
        if adapter is None:
            error = f"unknown broker source '{source_name}'"
            source_report["source_errors"] = [error]
            _mark_source_unavailable(symbol_reports, source_name, resolved_symbols, error)
            continue
        source_report["enabled"] = adapter.is_enabled()
        if not adapter.is_enabled():
            error = f"{source_name} disabled or missing credentials"
            source_report["source_errors"] = [error]
            _mark_source_unavailable(symbol_reports, source_name, resolved_symbols, error)
            continue

        quote_payloads: dict[str, dict] = {}
        bulk_quote_error = ""
        try:
            raw_quotes = adapter.get_quote(resolved_symbols)
            quote_payloads = raw_quotes if isinstance(raw_quotes, dict) else {}
        except Exception as exc:  # noqa: BLE001 - diagnostics must continue across sources
            bulk_quote_error = _redact_broker_error(exc, settings)
            source_report["source_errors"] = [bulk_quote_error]

        for symbol in resolved_symbols:
            payload = _broker_quote_payload(quote_payloads, symbol)
            ltp = _quote_price(payload)
            quote_ok = ltp > 0.0
            if quote_ok:
                source_report["quote_success"] = int(source_report["quote_success"]) + 1
            else:
                source_report["quote_failures"] = int(source_report["quote_failures"]) + 1

            view = {
                "enabled": True,
                "quote_ok": quote_ok,
                "historical_ok": False,
                "ltp": round(ltp, 4),
                "latest_bar_date": None,
                "latest_close": 0.0,
                "stale": False,
                "errors": [bulk_quote_error] if bulk_quote_error else [],
            }
            symbol_sources = cast(dict[str, dict[str, object]], symbol_reports[symbol]["sources"])
            symbol_sources[source_name] = view

        for symbol in resolved_symbols:
            symbol_sources = cast(dict[str, dict[str, object]], symbol_reports[symbol]["sources"])
            view = symbol_sources[source_name]
            try:
                rows = adapter.get_historical(symbol, interval, start, end)
            except Exception as exc:  # noqa: BLE001 - one bad symbol/source should not stop diagnostics
                view["errors"] = [*cast(list[str], view.get("errors", [])), _redact_broker_error(exc, settings)]
                source_report["historical_failures"] = int(source_report["historical_failures"]) + 1
                continue

            latest = _latest_broker_bar(rows)
            latest_date = _broker_bar_date(latest)
            latest_close = _safe_broker_float(_mapping(latest).get("close"))
            historical_ok = bool(rows) and latest_close > 0.0
            stale = historical_ok and latest_date is not None and latest_date < end
            view.update(
                {
                    "historical_ok": historical_ok,
                    "latest_bar_date": latest_date.isoformat() if latest_date else None,
                    "latest_close": round(latest_close, 4),
                    "stale": stale,
                }
            )
            if historical_ok:
                source_report["historical_success"] = int(source_report["historical_success"]) + 1
            else:
                source_report["historical_failures"] = int(source_report["historical_failures"]) + 1
            if stale:
                stale_symbols = cast(list[str], source_report["stale_symbols"])
                stale_symbols.append(symbol)

        _finalize_broker_source_report(source_report, len(resolved_symbols))

    reconciliation = _reconcile_broker_sources(symbol_reports, broker_sources, price_tolerance_pct)
    recommendations = _broker_health_recommendations(source_reports, reconciliation)
    passed = bool(resolved_symbols) and any(
        bool(report.get("enabled"))
        and (int(report.get("quote_success", 0)) > 0 or int(report.get("historical_success", 0)) > 0)
        for report in source_reports.values()
    )
    report = {
        "pipeline": "broker_health",
        "run_at": datetime.utcnow().isoformat() + "Z",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "interval": interval,
        "symbols_requested": len(resolved_symbols),
        "sources": [source for source, _ in broker_sources],
        "price_tolerance_pct": price_tolerance_pct,
        "passed": passed,
        "source_reports": source_reports,
        "reconciliation": reconciliation,
        "symbol_reports": list(symbol_reports.values()),
        "recommendations": recommendations,
    }
    LocalFileStorage(settings.storage.root_dir).save_json(
        report,
        filename="broker_health_report.json",
        subdir="quality",
    )
    return report


def run_screen(config_path: str | None = None) -> dict[str, object]:
    """Run the full market screening pass (daily + intraday) and return ranked signals."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)

    daily = run_daily(settings)
    intraday = run_intraday(settings)
    brokers = summarize_brokers(settings)
    features = cast(list[FeatureVector], daily.get("features", []))
    scores = cast(list[ScoreCard], daily.get("scores", []))
    long_signals = cast(list[SignalResult], daily.get("long_signals", []))
    swing_signals = cast(list[SignalResult], daily.get("swing_signals", []))
    short_signals_top = cast(list[SignalResult], daily.get("short_signals_top", []))
    intraday_swing_signals = cast(list[SignalResult], intraday.get("swing_signals", []))
    report_symbols = [fv.symbol for fv in features]
    company_metadata = _company_metadata_for_reports(settings, report_symbols)
    long_reports = build_signal_reports(
        features,
        scores,
        long_signals,
        signal_type="long_term",
        company_metadata=company_metadata,
        limit=10,
    )
    swing_reports = build_signal_reports(
        features,
        scores,
        swing_signals,
        signal_type="swing",
        company_metadata=company_metadata,
        limit=10,
    )
    sector_reports = SectorIntelligenceBuilder().build_from_engine_output(daily)

    return {
        "broker_enabled": brokers,
        "daily_top_long": [
            {
                "symbol": s.symbol,
                "sector": s.sector,
                "score": round(s.score, 2),
                "category": s.category,
                "conviction": round(s.explanation.confidence, 2),
                "horizon": s.explanation.holding_horizon,
                "top_drivers": s.explanation.top_positive_drivers[:3],
                "top_risks": s.explanation.top_negative_drivers[:2],
                "entry_logic": s.explanation.entry_logic,
            }
            for s in long_signals[:5]
        ],
        "daily_top_swing": [
            {
                "symbol": s.symbol,
                "sector": s.sector,
                "score": round(s.score, 2),
                "category": s.category,
                "conviction": round(s.explanation.confidence, 2),
                "horizon": s.explanation.holding_horizon,
                "top_drivers": s.explanation.top_positive_drivers[:3],
                "top_risks": s.explanation.top_negative_drivers[:2],
                "entry_logic": s.explanation.entry_logic,
            }
            for s in swing_signals[:5]
        ],
        "daily_top_short": [
            {
                "symbol": s.symbol,
                "sector": s.sector,
                "score": round(s.score, 2),
                "category": s.category,
                "conviction": round(s.explanation.confidence, 2),
                "horizon": s.explanation.holding_horizon,
                "top_drivers": s.explanation.top_positive_drivers[:3],
                "risk_amplifiers": s.explanation.top_negative_drivers[:2],
                "entry_logic": s.explanation.entry_logic,
                "invalidation_logic": s.explanation.invalidation_logic,
            }
            for s in short_signals_top[:5]
        ],
        "intraday_top_swing": [
            {
                "symbol": s.symbol,
                "sector": s.sector,
                "score": round(s.score, 2),
                "category": s.category,
                "conviction": round(s.explanation.confidence, 2),
            }
            for s in intraday_swing_signals[:5]
        ],
        "professional_signal_reports": {
            "long_term": [report.to_dict() for report in long_reports],
            "swing": [report.to_dict() for report in swing_reports],
            "console_rows": {
                "long_term": signal_reports_to_console_rows(long_reports),
                "swing": signal_reports_to_console_rows(swing_reports),
            },
            "markdown_top_long": render_signal_markdown(long_reports[0]) if long_reports else "",
            "markdown_top_swing": render_signal_markdown(swing_reports[0]) if swing_reports else "",
        },
        "sector_rankings": [report.to_dict() for report in sector_reports],
    }


# Keep the old name as an alias so existing tests and scripts don't break.
run_demo = run_screen


def run_single_stock(
    symbol: str,
    config_path: str | None = None,
) -> dict[str, object]:
    """Deep single-stock analysis: technicals, fundamentals, text signals, multi-horizon assessment."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)

    market = _build_market_provider(settings)
    financials = _build_financials_provider(settings)
    text = _build_text_provider(settings)

    # Always run the NLP/news pipeline for deep analysis, regardless of the
    # global nlp.enabled flag — it's the whole point of single-stock mode.
    nlp_on = replace(settings.nlp, enabled=True)
    settings_for_nlp = replace(settings, nlp=nlp_on)
    text_pipeline = _build_text_pipeline(settings_for_nlp, text)
    from stock_screener_engine.pipelines.single_stock_deep import SingleStockPipeline

    pipeline = SingleStockPipeline(
        settings=settings_for_nlp,
        market_data=market,
        text_data=text,
        financials=financials,
        text_pipeline=text_pipeline,
    )
    return pipeline.run(symbol)


def run_deepdive_report(
    symbol: str,
    config_path: str | None = None,
    document_path: str | None = None,
    output_format: str = "json",
) -> dict[str, object]:
    """Run company deep-dive analysis and assemble a research report."""
    analysis = run_single_stock(symbol=symbol, config_path=config_path)
    peer_insights = run_peer_report(
        symbol=symbol,
        config_path=config_path,
        as_of=date.fromisoformat(str(analysis.get("as_of"))),
        output_format="json",
    )
    document_insights: dict[str, object] | None = None
    if document_path:
        doc_result = run_document_ingest(
            symbol=symbol,
            file_path=document_path,
            config_path=config_path,
            company_name=str(analysis.get("company_name") or symbol.upper()),
            document_type="financial_report",
        )
        document_insights = doc_result

    builder = CompanyDeepDiveBuilder()
    report = builder.build(
        analysis,
        document_insights=document_insights,
        peer_insights=peer_insights,
    )
    payload = report.to_dict()
    payload["source_analysis"] = analysis
    payload["peer_comparison"] = peer_insights
    if document_insights is not None:
        payload["document_insights"] = document_insights
    if output_format == "markdown":
        payload["markdown"] = builder.render_markdown(report)
    return payload


def run_document_ingest(
    symbol: str,
    file_path: str,
    config_path: str | None = None,
    company_name: str | None = None,
    document_type: str = "unknown",
    publication_date: date | None = None,
    fiscal_period: str | None = None,
) -> dict[str, object]:
    """Ingest and analyze a local financial document."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)
    pipeline = DocumentIntelligencePipeline(settings=settings)
    result = pipeline.run(
        symbol=symbol.strip().upper(),
        file_path=file_path,
        company_name=company_name,
        document_type=document_type,
        publication_date=publication_date,
        fiscal_period=fiscal_period,
    )
    return result.to_dict()


def run_sector_rankings(config_path: str | None = None) -> dict[str, object]:
    """Run daily scan and return sector intelligence rankings."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)
    daily = run_daily(settings)
    reports = SectorIntelligenceBuilder().build_from_engine_output(daily)
    return {
        "as_of": daily.get("as_of"),
        "sector_rankings": [report.to_dict() for report in reports],
        "markdown": SectorIntelligenceBuilder().render_markdown(reports),
    }


def run_peer_report(
    symbol: str,
    config_path: str | None = None,
    as_of: date | None = None,
    output_format: str = "json",
) -> dict[str, object]:
    """Build a canonical sector peer-comparison report for one stock."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)

    from stock_screener_engine.data_sources.financials.sqlite_financials_provider import SQLiteFinancialsProvider

    report_as_of = as_of or date.today()
    store = MarketDataStore(settings.storage.sqlite_path)
    financials = SQLiteFinancialsProvider(
        sqlite_path=settings.storage.sqlite_path,
        venue=settings.runtime_data.canonical_venue,
        store=store,
    )
    try:
        report = PeerComparisonBuilder(
            store=store,
            financials=financials,
            venue=settings.runtime_data.canonical_venue,
        ).build(symbol, as_of=report_as_of)
        payload = report.to_dict()
        if output_format == "markdown":
            payload["markdown"] = render_peer_markdown(report)
        return payload
    finally:
        financials.close()


def run_sector_peer_report(
    sector: str,
    config_path: str | None = None,
    as_of: date | None = None,
    output_format: str = "json",
) -> dict[str, object]:
    """Build canonical peer rankings for all covered stocks in one sector."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)

    from stock_screener_engine.data_sources.financials.sqlite_financials_provider import SQLiteFinancialsProvider

    report_as_of = as_of or date.today()
    store = MarketDataStore(settings.storage.sqlite_path)
    financials = SQLiteFinancialsProvider(
        sqlite_path=settings.storage.sqlite_path,
        venue=settings.runtime_data.canonical_venue,
        store=store,
    )
    try:
        report = PeerComparisonBuilder(
            store=store,
            financials=financials,
            venue=settings.runtime_data.canonical_venue,
        ).build_sector(sector, as_of=report_as_of)
        payload = report.to_dict()
        if output_format == "markdown":
            payload["markdown"] = render_sector_peer_markdown(report)
        return payload
    finally:
        financials.close()


def run_data_foundation(
    start: date,
    end: date,
    symbols: list[str] | None = None,
    config_path: str | None = None,
    interval: str = "1d",
    universe_file: str | None = None,
) -> dict[str, object]:
    """Build the canonical security/calendar/OHLCV/corporate-action store."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)
    resolved_symbols, security_records = _resolve_runtime_universe(
        settings=settings,
        symbols=symbols,
        universe_file=universe_file,
    )
    pipeline = _build_data_foundation_pipeline(
        settings,
        security_master_records=security_records,
    )
    try:
        return pipeline.run(
            symbols=resolved_symbols,
            start=start,
            end=end,
            interval=interval,
        )
    finally:
        pipeline.close()


def run_data_quality(
    start: date,
    end: date,
    symbols: list[str] | None = None,
    config_path: str | None = None,
    interval: str = "1d",
    universe_file: str | None = None,
) -> dict[str, object]:
    """Read the canonical store and report data quality/reconciliation status."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)
    resolved_symbols, _ = _resolve_runtime_universe(
        settings=settings,
        symbols=symbols,
        universe_file=universe_file,
    )
    pipeline = _build_data_foundation_pipeline(settings, market_adapters=[])
    try:
        return pipeline.quality_report(
            symbols=resolved_symbols,
            start=start,
            end=end,
            interval=interval,
        )
    finally:
        pipeline.close()


def run_market_refresh(
    start: date,
    end: date,
    symbols: list[str] | None = None,
    config_path: str | None = None,
    interval: str = "1d",
    universe_file: str | None = None,
    batch_size: int = 25,
    retries: int = 2,
    retry_delay_seconds: float = 2.0,
    run_scan: bool = False,
    scan_mode: str = "swing",
) -> dict[str, object]:
    """Refresh canonical market data with retry/backoff and quality gates."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)
    resolved_symbols, security_records = _resolve_runtime_universe(
        settings=settings,
        symbols=symbols,
        universe_file=universe_file,
    )
    batch_size = max(1, int(batch_size))
    retries = max(0, int(retries))
    retry_delay_seconds = max(0.0, float(retry_delay_seconds))
    record_map = {record.symbol: record for record in security_records or []}
    attempts: list[dict[str, object]] = []
    pending_retry: set[str] = set()

    for batch_number, batch_symbols in enumerate(_chunks(resolved_symbols, batch_size), start=1):
        report = _run_data_foundation_attempt(
            settings=settings,
            symbols=batch_symbols,
            start=start,
            end=end,
            interval=interval,
            security_records=[record_map[symbol] for symbol in batch_symbols if symbol in record_map],
        )
        attempts.append(
            _refresh_attempt_summary(
                report,
                phase="initial",
                attempt=0,
                batch_number=batch_number,
                requested_symbols=batch_symbols,
            )
        )
        pending_retry.update(_symbols_needing_retry(report, batch_symbols))

    for attempt_number in range(1, retries + 1):
        if not pending_retry:
            break
        if retry_delay_seconds:
            time.sleep(retry_delay_seconds)
        retry_symbols = sorted(pending_retry)
        pending_retry.clear()
        for batch_number, batch_symbols in enumerate(_chunks(retry_symbols, batch_size), start=1):
            report = _run_data_foundation_attempt(
                settings=settings,
                symbols=batch_symbols,
                start=start,
                end=end,
                interval=interval,
                security_records=[record_map[symbol] for symbol in batch_symbols if symbol in record_map],
            )
            attempts.append(
                _refresh_attempt_summary(
                    report,
                    phase="retry",
                    attempt=attempt_number,
                    batch_number=batch_number,
                    requested_symbols=batch_symbols,
                )
            )
            pending_retry.update(_symbols_needing_retry(report, batch_symbols))

    normalization = _normalize_refresh_daily_bars(settings, resolved_symbols, interval)
    quality_report = _run_data_quality_attempt(
        settings=settings,
        symbols=resolved_symbols,
        start=start,
        end=end,
        interval=interval,
    )
    passed = bool(quality_report.get("passed")) and not pending_retry
    scan_summary = None
    if run_scan and passed:
        scan_summary = _run_refresh_scan_summary(resolved_symbols, scan_mode, config_path)

    report = {
        "pipeline": "market_refresh",
        "run_at": datetime.utcnow().isoformat() + "Z",
        "source": settings.runtime_data.market_provider,
        "canonical_venue": settings.runtime_data.canonical_venue,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "interval": interval,
        "symbols_requested": len(resolved_symbols),
        "batch_size": batch_size,
        "retries": retries,
        "retry_delay_seconds": retry_delay_seconds,
        "passed": passed,
        "failed_symbols": sorted(pending_retry),
        "attempts": attempts,
        "normalization": normalization,
        "quality": quality_report,
        "scan": scan_summary,
    }
    LocalFileStorage(settings.storage.root_dir).save_json(
        report,
        filename="market_refresh_report.json",
        subdir="quality",
    )
    if not passed:
        raise RuntimeError("Market refresh blocked by failed symbols or quality issues")
    return report


def _run_data_foundation_attempt(
    settings: AppSettings,
    symbols: Sequence[str],
    start: date,
    end: date,
    interval: str,
    security_records: Sequence[object],
) -> dict[str, object]:
    pipeline = _build_data_foundation_pipeline(
        settings,
        security_master_records=list(security_records),
    )
    try:
        return pipeline.run(
            symbols=list(symbols),
            start=start,
            end=end,
            interval=interval,
            raise_on_failure=False,
        )
    finally:
        pipeline.close()


def _run_data_quality_attempt(
    settings: AppSettings,
    symbols: Sequence[str],
    start: date,
    end: date,
    interval: str,
) -> dict[str, object]:
    pipeline = _build_data_foundation_pipeline(settings, market_adapters=[])
    try:
        return pipeline.quality_report(
            symbols=list(symbols),
            start=start,
            end=end,
            interval=interval,
        )
    finally:
        pipeline.close()


def _normalize_refresh_daily_bars(
    settings: AppSettings,
    symbols: Sequence[str],
    interval: str,
) -> dict[str, int]:
    store = MarketDataStore(settings.storage.sqlite_path)
    try:
        return store.normalize_daily_ohlcv(symbols=symbols, interval=interval)
    finally:
        store.close()


def _symbols_needing_retry(report: dict[str, object], requested_symbols: Sequence[str]) -> set[str]:
    requested = {symbol.strip().upper() for symbol in requested_symbols if symbol.strip()}
    retry: set[str] = set()

    for error in report.get("source_errors", []):
        if not isinstance(error, str):
            continue
        parts = error.split(":", maxsplit=2)
        if len(parts) >= 2:
            symbol = parts[1].strip().upper()
            if symbol in requested:
                retry.add(symbol)

    coverage = _mapping(report.get("coverage"))
    for symbol in coverage.get("missing_symbols", []):
        normalized = str(symbol).strip().upper()
        if normalized in requested:
            retry.add(normalized)

    quality_flags = _mapping(report.get("quality_flags"))
    ohlcv = _mapping(quality_flags.get("ohlcv"))
    for warning in ohlcv.get("warnings", []):
        if isinstance(warning, str) and warning.startswith("Missing OHLCV bars for:"):
            missing_text = warning.split(":", maxsplit=1)[1]
            for symbol in missing_text.split(","):
                normalized = symbol.strip().upper()
                if normalized in requested:
                    retry.add(normalized)

    reconciliation = _mapping(quality_flags.get("source_reconciliation"))
    for issue in reconciliation.get("issues", []):
        issue_map = _mapping(issue)
        if str(issue_map.get("severity", "")).lower() != "error":
            continue
        normalized = str(issue_map.get("symbol", "")).strip().upper()
        if normalized in requested:
            retry.add(normalized)

    return retry


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _resolve_broker_health_sources(sources: Sequence[str] | None) -> list[tuple[str, str]]:
    raw_sources = sources or ["zerodha", "icici_breeze"]
    resolved: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in raw_sources:
        adapter_key = _broker_adapter_key(raw)
        if not adapter_key or adapter_key in seen:
            continue
        seen.add(adapter_key)
        source_name = "icici_breeze" if adapter_key == "breeze" else adapter_key
        resolved.append((source_name, adapter_key))
    return resolved or [("zerodha", "zerodha"), ("icici_breeze", "breeze")]


def _broker_adapter_key(source: str) -> str:
    text = str(source or "").strip().lower().replace("-", "_")
    if text in {"icici", "icici_breeze", "breeze"}:
        return "breeze"
    if text in {"zerodha", "kite", "kiteconnect"}:
        return "zerodha"
    return text


def _new_broker_source_report(source_name: str, requested: int) -> dict[str, object]:
    return {
        "source": source_name,
        "enabled": False,
        "symbols_requested": requested,
        "quote_success": 0,
        "quote_failures": 0,
        "quote_coverage": 0.0,
        "historical_success": 0,
        "historical_failures": 0,
        "historical_coverage": 0.0,
        "stale_symbols": [],
        "source_errors": [],
    }


def _mark_source_unavailable(
    symbol_reports: Mapping[str, dict[str, object]],
    source_name: str,
    symbols: Sequence[str],
    error: str,
) -> None:
    for symbol in symbols:
        symbol_sources = cast(dict[str, dict[str, object]], symbol_reports[symbol]["sources"])
        symbol_sources[source_name] = {
            "enabled": False,
            "quote_ok": False,
            "historical_ok": False,
            "ltp": 0.0,
            "latest_bar_date": None,
            "latest_close": 0.0,
            "stale": False,
            "errors": [error],
        }


def _broker_quote_payload(quote_payloads: Mapping[str, object], symbol: str) -> dict:
    symbol = symbol.strip().upper()
    return _mapping(
        quote_payloads.get(symbol)
        or quote_payloads.get(f"NSE:{symbol}")
        or quote_payloads.get(symbol.replace("&", "%26"))
    )


def _quote_price(payload: Mapping[str, object]) -> float:
    for key in ("ltp", "last_price", "last", "close"):
        value = _safe_broker_float(payload.get(key))
        if value > 0.0:
            return value
    ohlc = _mapping(payload.get("ohlc"))
    return _safe_broker_float(ohlc.get("close"))


def _latest_broker_bar(rows: object) -> dict:
    if not isinstance(rows, list):
        return {}
    dated_rows = [(_broker_bar_date(row), row) for row in rows if isinstance(row, dict)]
    valid = [(bar_date, row) for bar_date, row in dated_rows if bar_date is not None]
    if valid:
        return max(valid, key=lambda item: item[0])[1]
    return rows[-1] if rows and isinstance(rows[-1], dict) else {}


def _broker_bar_date(row: object) -> date | None:
    value = _mapping(row).get("date") or _mapping(row).get("timestamp") or _mapping(row).get("datetime")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _safe_broker_float(value: object) -> float:
    try:
        return float(str(value).strip().replace(",", "")) if value is not None and str(value).strip() else 0.0
    except (TypeError, ValueError):
        return 0.0


def _finalize_broker_source_report(report: dict[str, object], requested: int) -> None:
    if requested <= 0:
        return
    report["quote_coverage"] = round(int(report["quote_success"]) / requested, 4)
    report["historical_coverage"] = round(int(report["historical_success"]) / requested, 4)


def _reconcile_broker_sources(
    symbol_reports: Mapping[str, dict[str, object]],
    broker_sources: Sequence[tuple[str, str]],
    price_tolerance_pct: float,
) -> dict[str, object]:
    price_mismatches: list[dict[str, object]] = []
    close_mismatches: list[dict[str, object]] = []
    preferred_counts: dict[str, int] = {}
    source_order = [source for source, _ in broker_sources]

    for symbol, report in symbol_reports.items():
        source_views = cast(dict[str, dict[str, object]], report["sources"])
        quote_prices = {
            source: _safe_broker_float(view.get("ltp"))
            for source, view in source_views.items()
            if bool(view.get("quote_ok")) and _safe_broker_float(view.get("ltp")) > 0.0
        }
        close_prices = {
            source: _safe_broker_float(view.get("latest_close"))
            for source, view in source_views.items()
            if bool(view.get("historical_ok")) and _safe_broker_float(view.get("latest_close")) > 0.0
        }
        quote_diff = _price_diff_pct(quote_prices.values())
        close_diff = _price_diff_pct(close_prices.values())
        report["quote_mismatch_pct"] = quote_diff
        report["historical_close_mismatch_pct"] = close_diff
        if quote_diff > price_tolerance_pct:
            price_mismatches.append({"symbol": symbol, "diff_pct": quote_diff, "prices": quote_prices})
        if close_diff > price_tolerance_pct:
            close_mismatches.append({"symbol": symbol, "diff_pct": close_diff, "prices": close_prices})

        preferred = _preferred_broker_source(source_views, source_order)
        report["preferred_source"] = preferred
        if preferred:
            preferred_counts[preferred] = preferred_counts.get(preferred, 0) + 1

    return {
        "price_mismatch_count": len(price_mismatches),
        "historical_close_mismatch_count": len(close_mismatches),
        "price_mismatches": price_mismatches,
        "historical_close_mismatches": close_mismatches,
        "preferred_source_counts": preferred_counts,
    }


def _price_diff_pct(values: object) -> float:
    prices = [float(value) for value in values if float(value) > 0.0]
    if len(prices) < 2:
        return 0.0
    low = min(prices)
    if low <= 0.0:
        return 0.0
    return round(((max(prices) - low) / low) * 100.0, 4)


def _preferred_broker_source(source_views: Mapping[str, dict[str, object]], source_order: Sequence[str]) -> str:
    best_source = ""
    best_score = -1
    for source in source_order:
        view = source_views.get(source)
        if not view or not bool(view.get("enabled")):
            continue
        score = 0
        score += 2 if bool(view.get("quote_ok")) else 0
        score += 2 if bool(view.get("historical_ok")) else 0
        score -= 1 if bool(view.get("stale")) else 0
        if score > best_score:
            best_score = score
            best_source = source
    return best_source


def _broker_health_recommendations(
    source_reports: Mapping[str, dict[str, object]],
    reconciliation: Mapping[str, object],
) -> list[str]:
    recommendations: list[str] = []
    for source, report in source_reports.items():
        if not bool(report.get("enabled")):
            recommendations.append(f"Enable or fix credentials for {source} before relying on it.")
            continue
        if float(report.get("quote_coverage", 0.0)) < 0.95:
            recommendations.append(f"Investigate quote coverage for {source}.")
        if float(report.get("historical_coverage", 0.0)) < 0.95:
            recommendations.append(f"Investigate historical coverage for {source}.")
        stale_symbols = cast(list[str], report.get("stale_symbols", []))
        if stale_symbols:
            recommendations.append(f"Review stale historical bars for {source}: {', '.join(stale_symbols[:10])}.")
    if int(reconciliation.get("price_mismatch_count", 0)) > 0:
        recommendations.append("Review broker quote price mismatches before using live prices for signals.")
    if int(reconciliation.get("historical_close_mismatch_count", 0)) > 0:
        recommendations.append("Review broker historical close mismatches before backtest/live reconciliation.")
    return recommendations or ["Broker health checks passed for the tested universe."]


def _redact_broker_error(exc: Exception, settings: AppSettings) -> str:
    text = str(exc) or exc.__class__.__name__
    for secret in _broker_secret_values(settings):
        if secret and len(secret) >= 4:
            text = text.replace(secret, "[redacted]")
    return text[:500]


def _broker_secret_values(settings: AppSettings) -> list[str]:
    values: list[str] = []
    for integration in (settings.integrations.zerodha, settings.integrations.breeze):
        for value in integration.credentials().values():
            if value:
                values.append(value)
    return values


def _refresh_attempt_summary(
    report: dict[str, object],
    phase: str,
    attempt: int,
    batch_number: int,
    requested_symbols: Sequence[str],
) -> dict[str, object]:
    coverage = _mapping(report.get("coverage"))
    rows_persisted = _mapping(report.get("rows_persisted"))
    return {
        "phase": phase,
        "attempt": attempt,
        "batch_number": batch_number,
        "passed": bool(report.get("passed")),
        "symbols_requested": report.get("symbols_requested"),
        "ohlcv_bars": rows_persisted.get("ohlcv_bars", 0),
        "coverage": coverage.get("coverage"),
        "missing_symbols": coverage.get("missing_symbols", []),
        "source_errors": report.get("source_errors", []),
        "retry_symbols": sorted(_symbols_needing_retry(report, requested_symbols)),
    }


def _run_refresh_scan_summary(symbols: Sequence[str], scan_mode: str, config_path: str | None) -> dict[str, object]:
    with _temporary_env(
        {
            "SSE_MARKET_PROVIDER": "canonical",
            "SSE_MARKET_UNIVERSE": ",".join(symbols),
        }
    ):
        result = run_screen(config_path=config_path)
    return {
        "mode": scan_mode,
        "daily_top_long": result.get("daily_top_long", []) if scan_mode in {"daily", "full"} else [],
        "daily_top_swing": result.get("daily_top_swing", []) if scan_mode in {"swing", "full"} else [],
        "intraday_top_swing": result.get("intraday_top_swing", []) if scan_mode in {"swing", "full"} else [],
        "broker_enabled": result.get("broker_enabled", {}),
    }


def _chunks(values: Sequence[str], size: int) -> Iterator[list[str]]:
    for idx in range(0, len(values), size):
        yield list(values[idx : idx + size])


@contextmanager
def _temporary_env(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_backtest_readiness(
    start: date,
    end: date,
    symbols: list[str] | None = None,
    config_path: str | None = None,
    interval: str = "1d",
    universe_file: str | None = None,
    min_history_years: float = 5.0,
    min_history_rows: int | None = None,
    horizons: list[int] | None = None,
    require_fundamentals: bool = False,
) -> dict[str, object]:
    """Check whether canonical data can support serious historical evaluation."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)
    resolved_symbols, _ = _resolve_runtime_universe(
        settings=settings,
        symbols=symbols,
        universe_file=universe_file,
    )
    pipeline = BacktestReadinessPipeline(settings=settings)
    try:
        return pipeline.run(
            symbols=resolved_symbols,
            start=start,
            end=end,
            interval=interval,
            horizons=horizons or [5, 20, 60],
            thresholds=BacktestReadinessThresholds(
                min_history_years=min_history_years,
                min_history_rows=min_history_rows,
                require_fundamentals=require_fundamentals,
            ),
        )
    finally:
        pipeline.close()


def run_forward_return_labels(
    start: date,
    end: date,
    symbols: list[str] | None = None,
    config_path: str | None = None,
    interval: str = "1d",
    universe_file: str | None = None,
    universe_policy: str = "current",
    min_history_rows: int = 1000,
    horizons: list[int] | None = None,
) -> dict[str, object]:
    """Generate forward-return labels from canonical OHLCV bars."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)
    resolved_symbols, _ = _resolve_runtime_universe(
        settings=settings,
        symbols=symbols,
        universe_file=universe_file,
    )
    pipeline = BacktestDatasetPipeline(settings=settings)
    try:
        return pipeline.build_forward_labels(
            symbols=resolved_symbols,
            start=start,
            end=end,
            horizons=horizons or [5, 20, 60],
            universe_policy=universe_policy,
            min_history_rows=min_history_rows,
            interval=interval,
        )
    finally:
        pipeline.close()


def run_technical_backtest(
    start: date,
    end: date,
    symbols: list[str] | None = None,
    config_path: str | None = None,
    interval: str = "1d",
    universe_file: str | None = None,
    universe_policy: str = "eligible_history",
    min_history_rows: int = 1000,
    min_lookback: int = 220,
    horizons: list[int] | None = None,
    round_trip_cost_bps: float | None = None,
    slippage_bps: float = 5.0,
) -> dict[str, object]:
    """Run first-pass technical/swing ranking evaluation on canonical OHLCV."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)
    resolved_symbols, _ = _resolve_runtime_universe(
        settings=settings,
        symbols=symbols,
        universe_file=universe_file,
    )
    pipeline = BacktestDatasetPipeline(settings=settings)
    try:
        return pipeline.evaluate_technical_ranking(
            symbols=resolved_symbols,
            start=start,
            end=end,
            horizons=horizons or [5, 20, 60],
            universe_policy=universe_policy,
            min_history_rows=min_history_rows,
            min_lookback=min_lookback,
            interval=interval,
            cost_model=_build_cost_model(round_trip_cost_bps, slippage_bps),
        )
    finally:
        pipeline.close()


def run_engine_backtest(
    start: date,
    end: date,
    symbols: list[str] | None = None,
    config_path: str | None = None,
    interval: str = "1d",
    universe_file: str | None = None,
    universe_policy: str = "eligible_history",
    min_history_rows: int = 1000,
    min_lookback: int = 220,
    horizons: list[int] | None = None,
    score_type: str = "swing",
    round_trip_cost_bps: float | None = None,
    slippage_bps: float = 5.0,
) -> dict[str, object]:
    """Run historical evaluation using the engine feature/scoring stack."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)
    resolved_symbols, _ = _resolve_runtime_universe(
        settings=settings,
        symbols=symbols,
        universe_file=universe_file,
    )
    pipeline = BacktestDatasetPipeline(settings=settings)
    try:
        return pipeline.evaluate_engine_scores(
            symbols=resolved_symbols,
            start=start,
            end=end,
            horizons=horizons or [5, 20, 60],
            universe_policy=universe_policy,
            min_history_rows=min_history_rows,
            min_lookback=min_lookback,
            interval=interval,
            score_type=score_type,
            cost_model=_build_cost_model(round_trip_cost_bps, slippage_bps),
        )
    finally:
        pipeline.close()


def run_security_master_ingest(
    file_path: str,
    config_path: str | None = None,
    venue: str | None = None,
) -> dict[str, object]:
    """Ingest canonical security master rows from a local CSV file."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)

    from stock_screener_engine.data_sources.security_master.csv_loader import load_security_master_csv

    csv_path = Path(file_path)
    canonical_venue = (venue or settings.runtime_data.canonical_venue).strip().upper()
    accepted = load_security_master_csv(str(csv_path), default_exchange=canonical_venue)
    rejected_rows = 0
    issues: list[dict[str, object]] = []

    persisted = 0
    if accepted:
        store = MarketDataStore(settings.storage.sqlite_path)
        try:
            persisted = store.upsert_security_master(accepted)
        finally:
            store.close()

    report = {
        "pipeline": "security_master_ingest",
        "venue": canonical_venue,
        "source_file": str(csv_path),
        "passed": bool(accepted) and not issues,
        "accepted": len(accepted),
        "rejected_rows": rejected_rows,
        "persisted": persisted,
        "quality_issues": issues,
    }
    LocalFileStorage(settings.storage.root_dir).save_json(
        report,
        filename="security_master_ingest_report.json",
        subdir="quality",
    )
    return report


def run_financials_ingest(
    symbol: str,
    file_path: str,
    as_of: date,
    config_path: str | None = None,
    venue: str | None = None,
) -> dict[str, object]:
    """Ingest point-in-time financial statement rows from a local CSV file."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)

    from stock_screener_engine.data_sources.financials.ingestion import FinancialStatementIngestor
    from stock_screener_engine.monitoring.factor_quality import FactorQualityValidator

    csv_path = Path(file_path)
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    canonical_venue = (venue or settings.runtime_data.canonical_venue).strip().upper()
    normalized_symbol = symbol.strip().upper()
    ingested = FinancialStatementIngestor().ingest_rows(
        rows=rows,
        venue=canonical_venue,
        symbol=normalized_symbol,
        as_of=as_of,
    )
    quality = FactorQualityValidator().validate(ingested.records, as_of=as_of)
    persisted = 0
    if quality.passed:
        store = MarketDataStore(settings.storage.sqlite_path)
        try:
            persisted = store.upsert_financial_statements(ingested.records)
        finally:
            store.close()

    report = {
        "pipeline": "financials_ingest",
        "symbol": normalized_symbol,
        "venue": canonical_venue,
        "as_of": as_of.isoformat(),
        "source_file": str(csv_path),
        "passed": quality.passed,
        "accepted": len(ingested.records),
        "rejected_rows": ingested.rejected_rows,
        "persisted": persisted,
        "quality_issues": [asdict(issue) for issue in quality.issues],
    }
    LocalFileStorage(settings.storage.root_dir).save_json(
        report,
        filename=f"{normalized_symbol}_financials_ingest_report.json",
        subdir="quality",
    )
    return report


def run_valuation_ingest(
    symbol: str,
    file_path: str,
    as_of: date,
    config_path: str | None = None,
    venue: str | None = None,
) -> dict[str, object]:
    """Ingest point-in-time market-cap/share-count rows from a local CSV file."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)

    from stock_screener_engine.data_sources.schemas import EquityValuationRecord

    csv_path = Path(file_path)
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    canonical_venue = (venue or settings.runtime_data.canonical_venue).strip().upper()
    normalized_symbol = symbol.strip().upper()
    accepted: list[EquityValuationRecord] = []
    rejected_rows = 0
    issues: list[dict[str, object]] = []
    for idx, row in enumerate(rows, start=2):
        try:
            row_as_of = _csv_date(row.get("as_of") or row.get("date"))
            market_cap = _csv_float(row.get("market_cap"))
            if row_as_of > as_of:
                rejected_rows += 1
                continue
            if market_cap <= 0:
                rejected_rows += 1
                issues.append({"row": idx, "severity": "error", "message": "market_cap must be positive"})
                continue
            accepted.append(
                EquityValuationRecord(
                    venue=canonical_venue,
                    symbol=normalized_symbol,
                    as_of=row_as_of,
                    market_cap=market_cap,
                    shares_outstanding=_csv_float(row.get("shares_outstanding")),
                    free_float_market_cap=_csv_float(row.get("free_float_market_cap")),
                    enterprise_value=_csv_float(row.get("enterprise_value")),
                    currency=str(row.get("currency") or "INR"),
                    source_id=str(row.get("source_id") or ""),
                )
            )
        except ValueError as exc:
            rejected_rows += 1
            issues.append({"row": idx, "severity": "error", "message": str(exc)})

    persisted = 0
    if accepted:
        store = MarketDataStore(settings.storage.sqlite_path)
        try:
            persisted = store.upsert_equity_valuations(accepted)
        finally:
            store.close()

    report = {
        "pipeline": "valuation_ingest",
        "symbol": normalized_symbol,
        "venue": canonical_venue,
        "as_of": as_of.isoformat(),
        "source_file": str(csv_path),
        "passed": bool(accepted) and not issues,
        "accepted": len(accepted),
        "rejected_rows": rejected_rows,
        "persisted": persisted,
        "quality_issues": issues,
    }
    LocalFileStorage(settings.storage.root_dir).save_json(
        report,
        filename=f"{normalized_symbol}_valuation_ingest_report.json",
        subdir="quality",
    )
    return report


def run_shareholding_ingest(
    symbol: str,
    file_path: str,
    as_of: date,
    config_path: str | None = None,
    venue: str | None = None,
) -> dict[str, object]:
    """Ingest point-in-time promoter/FII/DII/public shareholding rows from CSV."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)

    from stock_screener_engine.data_sources.schemas import ShareholdingRecord

    csv_path = Path(file_path)
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    canonical_venue = (venue or settings.runtime_data.canonical_venue).strip().upper()
    normalized_symbol = symbol.strip().upper()
    accepted: list[ShareholdingRecord] = []
    rejected_rows = 0
    issues: list[dict[str, object]] = []
    for idx, row in enumerate(rows, start=2):
        try:
            period_end = _csv_date(row.get("period_end"))
            filing_date = _csv_date(row.get("filing_date"))
            if period_end > as_of or filing_date > as_of:
                rejected_rows += 1
                continue
            promoter = _csv_float(row.get("promoter_pct"))
            fii = _csv_float(row.get("fii_pct"))
            dii = _csv_float(row.get("dii_pct"))
            public = _csv_float(row.get("public_pct"))
            if public == 0.0:
                public = max(0.0, 100.0 - promoter - fii - dii)
            values = [promoter, fii, dii, public]
            if any(value < 0.0 or value > 100.0 for value in values):
                rejected_rows += 1
                issues.append({"row": idx, "severity": "error", "message": "holding percentages must be between 0 and 100"})
                continue
            if sum(values) > 101.0:
                rejected_rows += 1
                issues.append({"row": idx, "severity": "error", "message": "holding percentages sum above 101"})
                continue
            accepted.append(
                ShareholdingRecord(
                    venue=canonical_venue,
                    symbol=normalized_symbol,
                    period_end=period_end,
                    filing_date=filing_date,
                    promoter_pct=promoter,
                    fii_pct=fii,
                    dii_pct=dii,
                    public_pct=public,
                    source_id=str(row.get("source_id") or ""),
                )
            )
        except ValueError as exc:
            rejected_rows += 1
            issues.append({"row": idx, "severity": "error", "message": str(exc)})

    persisted = 0
    if accepted:
        store = MarketDataStore(settings.storage.sqlite_path)
        try:
            persisted = store.upsert_shareholding(accepted)
        finally:
            store.close()

    report = {
        "pipeline": "shareholding_ingest",
        "symbol": normalized_symbol,
        "venue": canonical_venue,
        "as_of": as_of.isoformat(),
        "source_file": str(csv_path),
        "passed": bool(accepted) and not issues,
        "accepted": len(accepted),
        "rejected_rows": rejected_rows,
        "persisted": persisted,
        "quality_issues": issues,
    }
    LocalFileStorage(settings.storage.root_dir).save_json(
        report,
        filename=f"{normalized_symbol}_shareholding_ingest_report.json",
        subdir="quality",
    )
    return report


def run_factor_template(
    output_root: str,
    as_of: date,
    symbols: list[str] | None = None,
    universe_file: str | None = None,
    config_path: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Create external point-in-time factor CSV templates for a universe."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)
    resolved_symbols, _ = _resolve_runtime_universe(
        settings=settings,
        symbols=symbols,
        universe_file=universe_file,
    )
    pipeline = FactorBootstrapPipeline(settings=settings)
    try:
        return pipeline.create_templates(
            symbols=resolved_symbols,
            output_root=output_root,
            as_of=as_of,
            overwrite=overwrite,
        )
    finally:
        pipeline.close()


def run_factor_ingest(
    root: str,
    as_of: date,
    symbols: list[str] | None = None,
    universe_file: str | None = None,
    config_path: str | None = None,
    venue: str | None = None,
    min_coverage: float = 1.0,
) -> dict[str, object]:
    """Bulk ingest external point-in-time financial, valuation, and ownership factors."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)
    resolved_symbols, _ = _resolve_runtime_universe(
        settings=settings,
        symbols=symbols,
        universe_file=universe_file,
    )
    pipeline = FactorBootstrapPipeline(settings=settings)
    try:
        return pipeline.ingest(
            symbols=resolved_symbols,
            root=root,
            as_of=as_of,
            venue=venue,
            min_coverage=min_coverage,
        )
    finally:
        pipeline.close()


def _build_market_provider(settings: AppSettings):
    provider = settings.runtime_data.market_provider.strip().lower()
    if provider in {"canonical", "sqlite", "local", "local_sqlite", "market_store"}:
        from stock_screener_engine.data_sources.market.sqlite_market_data_provider import SQLiteMarketDataProvider

        return SQLiteMarketDataProvider(
            sqlite_path=settings.storage.sqlite_path,
            universe=settings.runtime_data.market_universe,
            venue=settings.runtime_data.canonical_venue,
            adjusted_history=settings.runtime_data.canonical_adjusted_history,
            strict_freshness=settings.runtime_data.canonical_strict_freshness,
            max_staleness_days=settings.runtime_data.canonical_max_staleness_days,
        )

    if provider in {"nse_http", "nse"}:
        from stock_screener_engine.data_sources.market.http_market_data_provider import NSEHTTPMarketDataProvider

        return NSEHTTPMarketDataProvider(universe=settings.runtime_data.market_universe)

    if provider == "yfinance":
        from stock_screener_engine.data_sources.market.yfinance_market_data_provider import YFinanceMarketDataProvider

        return YFinanceMarketDataProvider(universe=settings.runtime_data.market_universe)

    if provider in {"zerodha", "kite"}:
        from stock_screener_engine.data_sources.broker.zerodha_adapter import ZerodhaAdapter
        from stock_screener_engine.data_sources.market.broker_market_data_provider import BrokerMarketDataProvider

        return BrokerMarketDataProvider(
            broker=ZerodhaAdapter(settings.integrations.zerodha),
            universe=settings.runtime_data.market_universe,
            broker_name="zerodha",
            security_metadata=_load_canonical_security_metadata(settings),
        )

    if provider in {"icici", "breeze", "icici_breeze"}:
        from stock_screener_engine.data_sources.broker.breeze_adapter import BreezeAdapter
        from stock_screener_engine.data_sources.market.broker_market_data_provider import BrokerMarketDataProvider

        return BrokerMarketDataProvider(
            broker=BreezeAdapter(settings.integrations.breeze),
            universe=settings.runtime_data.market_universe,
            broker_name="icici_breeze",
            security_metadata=_load_canonical_security_metadata(settings),
        )

    if provider == "mock":
        from stock_screener_engine.data_sources.market.mock_market_data import MockIndianMarketDataProvider

        return MockIndianMarketDataProvider()

    raise ValueError(f"Unsupported market provider: {settings.runtime_data.market_provider}")


def _load_canonical_security_metadata(settings: AppSettings) -> dict[str, dict[str, object]]:
    sqlite_path = Path(settings.storage.sqlite_path)
    if not sqlite_path.exists():
        return {}
    store = MarketDataStore(settings.storage.sqlite_path)
    try:
        return store.company_metadata(settings.runtime_data.market_universe)
    finally:
        store.close()


def _build_financials_provider(settings: AppSettings):
    provider = settings.runtime_data.financials_provider.strip().lower()
    market_provider = settings.runtime_data.market_provider.strip().lower()
    if provider in {"", "none", "disabled"} and market_provider in {
        "canonical",
        "sqlite",
        "local",
        "local_sqlite",
        "market_store",
    }:
        provider = "canonical"
    if provider in {"", "none", "disabled"}:
        return None
    if provider in {"canonical", "sqlite", "local_sqlite", "market_store"}:
        from stock_screener_engine.data_sources.financials.sqlite_financials_provider import SQLiteFinancialsProvider

        return SQLiteFinancialsProvider(
            sqlite_path=settings.storage.sqlite_path,
            venue=settings.runtime_data.canonical_venue,
        )
    if provider == "mock":
        from stock_screener_engine.data_sources.market.mock_fundamentals import MockFinancialsProvider

        return MockFinancialsProvider()
    raise ValueError(f"Unsupported financials provider: {settings.runtime_data.financials_provider}")


def _company_metadata_for_reports(
    settings: AppSettings,
    symbols: list[str],
) -> dict[str, dict[str, object]]:
    if not symbols:
        return {}
    provider = _build_market_provider(settings)
    try:
        return _company_metadata_from_provider(provider, symbols)
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()


def _company_metadata_from_provider(provider: object, symbols: list[str]) -> dict[str, dict[str, object]]:
    metadata_loader = getattr(provider, "get_company_metadata", None)
    if not callable(metadata_loader):
        return {}
    metadata = metadata_loader(symbols)
    return metadata if isinstance(metadata, dict) else {}


def _build_data_foundation_pipeline(
    settings: AppSettings,
    market_adapters: list | None = None,
    security_master_records: list | None = None,
) -> DataFoundationPipeline:
    if market_adapters is None:
        from stock_screener_engine.data_sources.market.provider_ingestion_adapter import ProviderMarketIngestionAdapter

        market_provider = _build_market_provider(settings)
        market_adapters = [
            ProviderMarketIngestionAdapter(
                provider=market_provider,
                venue=settings.runtime_data.canonical_venue,
            )
        ]

    exchange_adapters = []
    provider = settings.runtime_data.market_provider.strip().lower()
    if provider in {"nse_http", "nse"}:
        from stock_screener_engine.data_sources.exchange.nse_http_adapter import NSEHTTPAdapter

        exchange_adapters.append(NSEHTTPAdapter())

    return DataFoundationPipeline(
        settings=settings,
        market_adapters=market_adapters,
        exchange_adapters=exchange_adapters,
        security_master_records=security_master_records,
    )


def _resolve_runtime_universe(
    settings: AppSettings,
    symbols: list[str] | None = None,
    universe_file: str | None = None,
) -> tuple[list[str], list | None]:
    file_records = _load_universe_records(
        universe_file=universe_file,
        venue=settings.runtime_data.canonical_venue,
    )
    if symbols:
        resolved = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
    elif file_records:
        resolved = [record.symbol for record in file_records]
    else:
        resolved = settings.runtime_data.market_universe

    if not file_records:
        return resolved, None

    by_symbol = {record.symbol: record for record in file_records}
    from stock_screener_engine.data_sources.security_master.provider import build_minimal_security_master

    fallback = {
        record.symbol: record
        for record in build_minimal_security_master(
            [symbol for symbol in resolved if symbol not in by_symbol],
            exchange=settings.runtime_data.canonical_venue,
        )
    }
    combined = {**fallback, **by_symbol}
    return resolved, [combined[symbol] for symbol in resolved if symbol in combined]


def _load_universe_records(universe_file: str | None, venue: str) -> list:
    if not universe_file:
        return []
    from stock_screener_engine.data_sources.security_master.csv_loader import load_security_master_csv

    return load_security_master_csv(universe_file, default_exchange=venue)


def _build_cost_model(round_trip_cost_bps: float | None, slippage_bps: float) -> IndianEquityCostModel:
    return IndianEquityCostModel(
        explicit_round_trip_bps=round_trip_cost_bps,
        slippage_bps_per_side=slippage_bps,
    )


def _build_text_provider(settings: AppSettings) -> FreeRSSNewsProvider:
    provider = settings.runtime_data.news_provider.strip().lower()
    if provider not in {"free_rss", "google_news_rss"}:
        raise ValueError(f"Unsupported news provider: {settings.runtime_data.news_provider}")
    return FreeRSSNewsProvider()


def _build_text_pipeline(settings: AppSettings, text: FreeRSSNewsProvider) -> TextIntelligencePipeline | None:
    if not settings.nlp.enabled:
        return None

    filings_provider_name = settings.runtime_data.filings_provider.strip().lower()
    filings_provider = (
        ExchangeFilingsProvider()
        if filings_provider_name in {"exchange_announcements", "nse", "nse_announcements"}
        else NullFilingsProvider()
    )
    transcript_provider = NullTranscriptProvider()
    adapters = [
        GenericNewsAdapter(text),
        FilingsAdapter(filings_provider),
        TranscriptsAdapter(transcript_provider),
    ]
    llm_config = settings.llm
    if not llm_config.enabled:
        llm_config = replace(llm_config, provider="heuristic", model="heuristic-finance-v1")

    llm_client = build_llm_client(llm_config)

    return TextIntelligencePipeline(
        ingestor=TextDocumentIngestor(
            adapters=adapters,
            health_sink=IngestionHealthSink(settings.storage.root_dir),
        ),
        aggregator=EventFeatureAggregator(
            half_life_days=settings.nlp.decay_half_life_days,
            high_impact_threshold=settings.nlp.high_impact_threshold,
        ),
        enable_sentiment=settings.nlp.enable_sentiment,
        enable_event_extraction=settings.nlp.enable_event_extraction,
        llm_enabled=settings.llm.enabled,
        llm_min_confidence=settings.llm.min_confidence,
        llm_fallback_to_rules=settings.llm.fallback_to_rules,
        llm_provider_name=settings.llm.provider,
        llm_model_name=settings.llm.model,
        audit_low_confidence=settings.llm.audit_low_confidence,
        audit_sink=LowConfidenceAuditSink(settings.llm.audit_path),
        llm_classifier=LLMDocumentClassifier(llm_client),
        llm_event_extractor=LLMEventExtractor(llm_client),
        llm_sentiment_extractor=LLMSentimentExtractor(llm_client),
        llm_management_tone_extractor=(LLMManagementToneExtractor(llm_client) if settings.llm.enable_management_tone else None),
    )


def _csv_date(value: object) -> date:
    if isinstance(value, date):
        return value
    if value is None or not str(value).strip():
        raise ValueError("date/as_of is required")
    return date.fromisoformat(str(value).strip())


def _csv_optional_date(value: object) -> date | None:
    if value is None or not str(value).strip():
        return None
    return _csv_date(value)


def _csv_float(value: object) -> float:
    if value is None or str(value).strip() == "":
        return 0.0
    return float(str(value).strip().replace(",", ""))


def _csv_int(value: object, default: int = 0) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(float(str(value).strip().replace(",", "")))


def _csv_bool(value: object, default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "active"}:
        return True
    if text in {"0", "false", "no", "n", "inactive", "delisted"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")
