from __future__ import annotations

from stock_screener_engine.config.settings import load_settings
from stock_screener_engine.core.engine import ResearchEngine
from stock_screener_engine.data_sources.filings.filings_adapter import FilingsAdapter
from stock_screener_engine.data_sources.filings.mock_filings import MockFilingsProvider
from stock_screener_engine.data_sources.market.mock_fundamentals import MockFinancialsProvider
from stock_screener_engine.data_sources.market.mock_market_data import MockIndianMarketDataProvider
from stock_screener_engine.data_sources.news.generic_news_adapter import GenericNewsAdapter
from stock_screener_engine.data_sources.news.mock_news import MockNewsProvider
from stock_screener_engine.data_sources.text.mock_text_adapter import MockTextEventProvider
from stock_screener_engine.data_sources.transcripts.mock_transcripts import MockTranscriptProvider
from stock_screener_engine.data_sources.transcripts.transcripts_adapter import TranscriptsAdapter
from stock_screener_engine.nlp.event_engine.pipeline import TextIntelligencePipeline
from stock_screener_engine.nlp.ingestion.document_ingestor import TextDocumentIngestor
from stock_screener_engine.nlp.ingestion.text_event_adapter import TextEventAdapter


def test_research_engine_emits_text_features_when_pipeline_enabled() -> None:
    settings = load_settings()
    settings = settings.__class__(
        environment=settings.environment,
        log_level=settings.log_level,
        storage=settings.storage,
        features=settings.features,
        nlp=settings.nlp.__class__(
            enabled=True,
            enable_sentiment=True,
            enable_event_extraction=True,
            lookback_days=14,
            decay_half_life_days=settings.nlp.decay_half_life_days,
            high_impact_threshold=settings.nlp.high_impact_threshold,
        ),
        llm=settings.llm.__class__(
            enabled=True,
            provider=settings.llm.provider,
            model=settings.llm.model,
            min_confidence=0.4,
            fallback_to_rules=True,
            enable_management_tone=True,
            audit_low_confidence=settings.llm.audit_low_confidence,
        ),
        runtime_data=settings.runtime_data,
        integrations=settings.integrations,
        scoring=settings.scoring,
    )

    text_provider = MockTextEventProvider()
    text_pipeline = TextIntelligencePipeline(
        ingestor=TextDocumentIngestor(
            adapters=[
                TextEventAdapter(text_provider),
                GenericNewsAdapter(MockNewsProvider()),
                FilingsAdapter(MockFilingsProvider()),
                TranscriptsAdapter(MockTranscriptProvider()),
            ]
        )
    )

    engine = ResearchEngine(
        settings=settings,
        market_data=MockIndianMarketDataProvider(),
        text_data=text_provider,
        financials=MockFinancialsProvider(),
        text_pipeline=text_pipeline,
    )
    out = engine.run(symbols=["RELIANCE", "TCS"], regime_score=0.2)
    assert out["text_features"]
    assert any("event_strength_score" in row for row in out["text_features"])
