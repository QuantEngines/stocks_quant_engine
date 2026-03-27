from __future__ import annotations

from datetime import datetime

from stock_screener_engine.llm.base.llm_client import LLMClient
from stock_screener_engine.llm.extraction.document_classifier import LLMDocumentClassifier
from stock_screener_engine.llm.extraction.event_extractor import LLMEventExtractor
from stock_screener_engine.nlp.event_engine.pipeline import TextIntelligencePipeline
from stock_screener_engine.nlp.ingestion.document_ingestor import TextDocumentIngestor
from stock_screener_engine.nlp.ingestion.text_event_adapter import TextEventAdapter
from stock_screener_engine.data_sources.text.mock_text_adapter import MockTextEventProvider


class BrokenClient(LLMClient):
    def complete(self, task: str, prompt: str, payload: dict[str, object]) -> dict[str, object]:
        return {"bad": "payload"}


def test_llm_pipeline_falls_back_to_rules_on_invalid_output() -> None:
    text = MockTextEventProvider()
    pipe = TextIntelligencePipeline(
        ingestor=TextDocumentIngestor(adapters=[TextEventAdapter(text)]),
        llm_enabled=True,
        llm_min_confidence=0.9,
        llm_fallback_to_rules=True,
        llm_classifier=LLMDocumentClassifier(BrokenClient()),
        llm_event_extractor=LLMEventExtractor(BrokenClient()),
    )
    out = pipe.run(symbols=["RELIANCE"], as_of=datetime.now(), lookback_days=7)
    assert "RELIANCE" in out.vectors
    assert out.vectors["RELIANCE"].recent_event_count >= 0.0
