"""LLM event aggregation wrapper around core NLP event aggregator."""

from __future__ import annotations

from datetime import datetime

from stock_screener_engine.nlp.event_engine.aggregation import EventFeatureAggregator
from stock_screener_engine.nlp.schemas.events import DocumentAnalysis, TextFeatureSet


class LLMEventAggregator:
    def __init__(self, half_life_days: float = 14.0, high_impact_threshold: float = 0.65) -> None:
        self._inner = EventFeatureAggregator(
            half_life_days=half_life_days,
            high_impact_threshold=high_impact_threshold,
        )

    def aggregate(self, analyses: list[DocumentAnalysis], as_of: datetime) -> TextFeatureSet:
        return self._inner.aggregate(analyses, as_of=as_of)
