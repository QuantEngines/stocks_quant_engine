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
from stock_screener_engine.llm.base.llm_client import HeuristicLLMClient
from stock_screener_engine.llm.extraction.document_classifier import LLMDocumentClassifier
from stock_screener_engine.llm.extraction.event_extractor import LLMEventExtractor
from stock_screener_engine.llm.extraction.management_tone_extractor import LLMManagementToneExtractor
from stock_screener_engine.llm.extraction.sentiment_extractor import LLMSentimentExtractor
from stock_screener_engine.nlp.event_engine.pipeline import TextIntelligencePipeline
from stock_screener_engine.nlp.ingestion.document_ingestor import TextDocumentIngestor
from stock_screener_engine.nlp.ingestion.text_event_adapter import TextEventAdapter


def _pipeline(llm_enabled: bool) -> TextIntelligencePipeline:
    text_provider = MockTextEventProvider()
    client = HeuristicLLMClient()
    return TextIntelligencePipeline(
        ingestor=TextDocumentIngestor(
            adapters=[
                TextEventAdapter(text_provider),
                GenericNewsAdapter(MockNewsProvider()),
                FilingsAdapter(MockFilingsProvider()),
                TranscriptsAdapter(MockTranscriptProvider()),
            ]
        ),
        llm_enabled=llm_enabled,
        llm_min_confidence=0.5,
        llm_fallback_to_rules=True,
        llm_classifier=LLMDocumentClassifier(client),
        llm_event_extractor=LLMEventExtractor(client),
        llm_sentiment_extractor=LLMSentimentExtractor(client),
        llm_management_tone_extractor=LLMManagementToneExtractor(client),
    )


def main() -> None:
    symbols = ["RELIANCE", "TCS", "INFY", "LT", "SBIN"]
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
            lookback_days=30,
            decay_half_life_days=settings.nlp.decay_half_life_days,
            high_impact_threshold=settings.nlp.high_impact_threshold,
        ),
        llm=settings.llm.__class__(
            enabled=True,
            provider="heuristic",
            model="heuristic-finance-v1",
            min_confidence=0.5,
            fallback_to_rules=True,
            enable_management_tone=True,
            audit_low_confidence=True,
        ),
        runtime_data=settings.runtime_data,
        integrations=settings.integrations,
        scoring=settings.scoring,
    )

    market = MockIndianMarketDataProvider()
    financials = MockFinancialsProvider()
    text_provider = MockTextEventProvider()

    engine_without_llm = ResearchEngine(
        settings=settings.__class__(
            environment=settings.environment,
            log_level=settings.log_level,
            storage=settings.storage,
            features=settings.features,
            nlp=settings.nlp,
            llm=settings.llm.__class__(
                enabled=False,
                provider=settings.llm.provider,
                model=settings.llm.model,
                min_confidence=settings.llm.min_confidence,
                fallback_to_rules=settings.llm.fallback_to_rules,
                enable_management_tone=settings.llm.enable_management_tone,
                audit_low_confidence=settings.llm.audit_low_confidence,
            ),
            runtime_data=settings.runtime_data,
            integrations=settings.integrations,
            scoring=settings.scoring,
        ),
        market_data=market,
        text_data=text_provider,
        financials=financials,
        text_pipeline=_pipeline(llm_enabled=False),
    )
    engine_with_llm = ResearchEngine(
        settings=settings,
        market_data=market,
        text_data=text_provider,
        financials=financials,
        text_pipeline=_pipeline(llm_enabled=True),
    )

    base = engine_without_llm.run(symbols=symbols, regime_score=0.2)
    enhanced = engine_with_llm.run(symbols=symbols, regime_score=0.2)

    print("\n=== LLM Event Intelligence Demo ===")
    print("Symbols:", ", ".join(symbols))

    print("\nSample structured text features (LLM enabled):")
    for row in enhanced["text_features"][:5]:
        print(row)

    print("\nScore impact (long-term / swing):")
    base_map = {s.symbol: s for s in base["scores"]}
    enhanced_map = {s.symbol: s for s in enhanced["scores"]}
    for sym in symbols:
        b = base_map[sym]
        e = enhanced_map[sym]
        print(
            f"{sym}: long {b.long_term_score:.2f} -> {e.long_term_score:.2f} | "
            f"swing {b.swing_score:.2f} -> {e.swing_score:.2f}"
        )


if __name__ == "__main__":
    main()
