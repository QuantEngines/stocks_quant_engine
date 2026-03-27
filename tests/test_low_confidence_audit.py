from __future__ import annotations

from datetime import datetime

from stock_screener_engine.data_sources.text.mock_text_adapter import MockTextEventProvider
from stock_screener_engine.llm.base.llm_client import HeuristicLLMClient
from stock_screener_engine.llm.extraction.document_classifier import LLMDocumentClassifier
from stock_screener_engine.llm.extraction.event_extractor import LLMEventExtractor
from stock_screener_engine.llm.extraction.sentiment_extractor import LLMSentimentExtractor
from stock_screener_engine.nlp.event_engine.audit import LowConfidenceAuditSink
from stock_screener_engine.nlp.event_engine.pipeline import TextIntelligencePipeline
from stock_screener_engine.nlp.ingestion.document_ingestor import TextDocumentIngestor
from stock_screener_engine.nlp.ingestion.text_event_adapter import TextEventAdapter


def test_low_confidence_outputs_are_persisted(tmp_path) -> None:
    provider = MockTextEventProvider()
    client = HeuristicLLMClient()

    pipeline = TextIntelligencePipeline(
        ingestor=TextDocumentIngestor(adapters=[TextEventAdapter(provider)]),
        llm_enabled=True,
        llm_min_confidence=0.99,
        llm_fallback_to_rules=True,
        llm_provider_name="heuristic",
        llm_model_name="heuristic-finance-v1",
        audit_low_confidence=True,
        audit_sink=LowConfidenceAuditSink(str(tmp_path)),
        llm_classifier=LLMDocumentClassifier(client),
        llm_event_extractor=LLMEventExtractor(client),
        llm_sentiment_extractor=LLMSentimentExtractor(client),
    )

    pipeline.run(symbols=["RELIANCE"], as_of=datetime.utcnow(), lookback_days=5)

    audit_files = list((tmp_path / "llm_audit").glob("*/low_confidence.jsonl"))
    assert audit_files, "expected low-confidence audit file"

    content = audit_files[0].read_text(encoding="utf-8")
    assert "\"task\":\"classification\"" in content
    assert "\"provider\":\"heuristic\"" in content
    assert "\"used_fallback\":true" in content
