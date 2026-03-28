"""Application entry helpers for pipeline execution."""

from __future__ import annotations

import logging
from dataclasses import replace

from stock_screener_engine.config.settings import AppSettings, load_settings
from stock_screener_engine.config.startup_validation import validate_startup_settings
from stock_screener_engine.data_sources.broker.factory import build_broker_adapters
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
from stock_screener_engine.pipelines.intraday_update import IntradayUpdatePipeline
from stock_screener_engine.pipelines.live_invalidation_daily import run_live_invalidation_daily_job


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_daily(settings: AppSettings) -> dict[str, list]:
    validate_startup_settings(settings)
    market = _build_market_provider(settings)
    text = _build_text_provider(settings)
    text_pipeline = _build_text_pipeline(settings, text)
    pipeline = DailyBatchPipeline(
        settings=settings,
        market_data=market,
        text_data=text,
        financials=None,
        text_pipeline=text_pipeline,
    )
    try:
        return pipeline.run()
    finally:
        pipeline.close()


def run_intraday(settings: AppSettings) -> dict[str, list]:
    validate_startup_settings(settings)
    market = _build_market_provider(settings)
    text = _build_text_provider(settings)
    text_pipeline = _build_text_pipeline(settings, text)
    pipeline = IntradayUpdatePipeline(
        settings=settings,
        market_data=market,
        text_data=text,
        financials=None,
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
        text_pipeline=text_pipeline,
    )
    return pipeline.run(symbol)


def _build_market_provider(settings: AppSettings):
    provider = settings.runtime_data.market_provider.strip().lower()
    if provider not in {"yfinance", "nse_http", "nse"}:
        raise ValueError(f"Unsupported market provider: {settings.runtime_data.market_provider}")
    from stock_screener_engine.data_sources.market.yfinance_market_data_provider import YFinanceMarketDataProvider

    return YFinanceMarketDataProvider(universe=settings.runtime_data.market_universe)


def _build_text_provider(settings: AppSettings) -> FreeRSSNewsProvider:
    provider = settings.runtime_data.news_provider.strip().lower()
    if provider not in {"free_rss", "google_news_rss"}:
        raise ValueError(f"Unsupported news provider: {settings.runtime_data.news_provider}")
    return FreeRSSNewsProvider()


def _build_text_pipeline(settings: AppSettings, text: FreeRSSNewsProvider) -> TextIntelligencePipeline | None:
    if not settings.nlp.enabled:
        return None

    filings_provider = NullFilingsProvider()
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
