from __future__ import annotations

from datetime import datetime

from stock_screener_engine.nlp.classification.rule_classifier import RuleDocumentClassifier
from stock_screener_engine.nlp.extraction.rule_event_extractor import RuleEventExtractor
from stock_screener_engine.nlp.schemas.events import DocumentCategory, EventType, NormalizedDocument, SourceType


def test_rule_classifier_detects_earnings() -> None:
    doc = NormalizedDocument(
        id="1",
        source=SourceType.NEWS,
        timestamp=datetime.utcnow(),
        symbol="ABC",
        title="ABC reports quarterly earnings beat",
        body_text="Margins improved and guidance raised",
        metadata={},
    )
    category = RuleDocumentClassifier().classify(doc)
    assert category == DocumentCategory.EARNINGS_RELATED


def test_rule_event_extractor_detects_order_win() -> None:
    doc = NormalizedDocument(
        id="2",
        source=SourceType.NEWS,
        timestamp=datetime.utcnow(),
        symbol="XYZ",
        title="XYZ wins large government order worth 500 crore",
        body_text="Order execution to start next quarter",
        metadata={},
    )
    event = RuleEventExtractor().extract(doc, time_decay_factor=1.0, entities={"company": ["XYZ"], "keywords": ["order"]})
    assert event.event_type == EventType.ORDER_WIN
    assert event.event_strength > 0.4
