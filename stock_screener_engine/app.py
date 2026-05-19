"""Application entry helpers for pipeline execution."""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

from stock_screener_engine.config.settings import AppSettings, load_settings
from stock_screener_engine.config.startup_validation import validate_startup_settings
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
from stock_screener_engine.pipelines.daily_batch import DailyBatchPipeline
from stock_screener_engine.pipelines.data_foundation import DataFoundationPipeline
from stock_screener_engine.pipelines.document_pipeline import DocumentIntelligencePipeline
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


def run_daily(settings: AppSettings) -> dict[str, list]:
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


def run_intraday(settings: AppSettings) -> dict[str, list]:
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


def run_screen(config_path: str | None = None) -> dict[str, object]:
    """Run the full market screening pass (daily + intraday) and return ranked signals."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)

    daily = run_daily(settings)
    intraday = run_intraday(settings)
    brokers = summarize_brokers(settings)
    report_symbols = [
        fv.symbol for fv in daily.get("features", [])
        if hasattr(fv, "symbol")
    ]
    company_metadata = _company_metadata_for_reports(settings, report_symbols)
    long_reports = build_signal_reports(
        daily["features"],
        daily["scores"],
        daily["long_signals"],
        signal_type="long_term",
        company_metadata=company_metadata,
        limit=10,
    )
    swing_reports = build_signal_reports(
        daily["features"],
        daily["scores"],
        daily["swing_signals"],
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
            for s in daily["long_signals"][:5]
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
            for s in daily["swing_signals"][:5]
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
            for s in daily.get("short_signals_top", [])[:5]
        ],
        "intraday_top_swing": [
            {
                "symbol": s.symbol,
                "sector": s.sector,
                "score": round(s.score, 2),
                "category": s.category,
                "conviction": round(s.explanation.confidence, 2),
            }
            for s in intraday["swing_signals"][:5]
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
) -> dict[str, object]:
    """Build the canonical security/calendar/OHLCV/corporate-action store."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)
    pipeline = _build_data_foundation_pipeline(settings)
    try:
        return pipeline.run(
            symbols=symbols or settings.runtime_data.market_universe,
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
) -> dict[str, object]:
    """Read the canonical store and report data quality/reconciliation status."""
    settings = load_settings(config_path=config_path)
    validate_startup_settings(settings)
    configure_logging(settings.log_level)
    pipeline = _build_data_foundation_pipeline(settings, market_adapters=[])
    try:
        return pipeline.quality_report(
            symbols=symbols or settings.runtime_data.market_universe,
            start=start,
            end=end,
            interval=interval,
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

    from stock_screener_engine.data_sources.schemas import SecurityMasterRecord

    csv_path = Path(file_path)
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    canonical_venue = (venue or settings.runtime_data.canonical_venue).strip().upper()
    accepted: list[SecurityMasterRecord] = []
    rejected_rows = 0
    issues: list[dict[str, object]] = []
    for idx, row in enumerate(rows, start=2):
        try:
            symbol = str(row.get("symbol") or row.get("tradingsymbol") or "").strip().upper()
            if not symbol:
                rejected_rows += 1
                issues.append({"row": idx, "severity": "error", "message": "symbol is required"})
                continue
            exchange = str(row.get("exchange") or canonical_venue).strip().upper()
            accepted.append(
                SecurityMasterRecord(
                    symbol=symbol,
                    exchange=exchange,
                    isin=str(row.get("isin") or "").strip(),
                    series=str(row.get("series") or "EQ").strip() or "EQ",
                    company_name=str(row.get("company_name") or row.get("name") or "").strip(),
                    sector=str(row.get("sector") or "Unknown").strip() or "Unknown",
                    industry=str(row.get("industry") or "Unknown").strip() or "Unknown",
                    listing_date=_csv_optional_date(row.get("listing_date")),
                    delisting_date=_csv_optional_date(row.get("delisting_date")),
                    active=_csv_bool(row.get("active"), default=True),
                    lot_size=_csv_int(row.get("lot_size"), default=1),
                    tick_size=_csv_float(row.get("tick_size") or 0.05),
                    source=str(row.get("source") or row.get("source_id") or "csv").strip() or "csv",
                )
            )
        except ValueError as exc:
            rejected_rows += 1
            issues.append({"row": idx, "severity": "error", "message": str(exc)})

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
        )

    if provider in {"icici", "breeze", "icici_breeze"}:
        from stock_screener_engine.data_sources.broker.breeze_adapter import BreezeAdapter
        from stock_screener_engine.data_sources.market.broker_market_data_provider import BrokerMarketDataProvider

        return BrokerMarketDataProvider(
            broker=BreezeAdapter(settings.integrations.breeze),
            universe=settings.runtime_data.market_universe,
            broker_name="icici_breeze",
        )

    if provider == "mock":
        from stock_screener_engine.data_sources.market.mock_market_data import MockIndianMarketDataProvider

        return MockIndianMarketDataProvider()

    raise ValueError(f"Unsupported market provider: {settings.runtime_data.market_provider}")


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
