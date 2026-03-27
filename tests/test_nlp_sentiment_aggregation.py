from __future__ import annotations

from datetime import datetime, timedelta

from stock_screener_engine.nlp.event_engine.aggregation import EventFeatureAggregator
from stock_screener_engine.nlp.schemas.events import (
    DocumentAnalysis,
    DocumentCategory,
    EventType,
    ExpectedDirection,
    ExtractedEvent,
    SentimentSignal,
    SentimentType,
    SourceType,
)
from stock_screener_engine.nlp.sentiment.rule_sentiment import RuleSentimentEngine
from stock_screener_engine.nlp.schemas.events import NormalizedDocument


def test_rule_sentiment_returns_domain_categories() -> None:
    doc = NormalizedDocument(
        id="s1",
        source=SourceType.TRANSCRIPT,
        timestamp=datetime.utcnow(),
        symbol="ABC",
        title="Management optimistic, guidance raised",
        body_text="Strong earnings and margin expansion expected",
        metadata={},
    )
    sentiments = RuleSentimentEngine().analyze(doc)
    types = {s.sentiment_type for s in sentiments}
    assert SentimentType.OUTLOOK in types
    assert SentimentType.EARNINGS in types


def test_event_aggregation_produces_numeric_feature_vector() -> None:
    now = datetime.utcnow()
    analysis = DocumentAnalysis(
        document_id="d1",
        symbol="ABC",
        category=DocumentCategory.CORPORATE_ACTION,
        event=ExtractedEvent(
            event_type=EventType.ORDER_WIN,
            timestamp=now - timedelta(days=1),
            symbol="ABC",
            confidence=0.8,
            source_type=SourceType.NEWS,
            summary="ABC wins order",
            entities={"company": ["ABC"], "keywords": ["order"]},
            event_strength=0.8,
            expected_direction=ExpectedDirection.POSITIVE,
            time_decay_factor=0.9,
            uncertainty_score=0.1,
        ),
        sentiments=[
            SentimentSignal(
                polarity_score=0.6,
                confidence=0.8,
                sentiment_type=SentimentType.OUTLOOK,
                intensity=0.7,
            )
        ],
    )
    feature_set = EventFeatureAggregator().aggregate([analysis], as_of=now)
    row = feature_set.vectors["ABC"]
    assert 0.0 <= row.event_strength_score <= 1.0
    assert -1.0 <= row.sentiment_score_recent <= 1.0
    assert row.catalyst_presence_flag >= 0.0
