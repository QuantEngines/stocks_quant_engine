"""LLM-assisted event extraction with normalized structured output."""

from __future__ import annotations

from stock_screener_engine.llm.base.llm_client import LLMClient
from stock_screener_engine.llm.base.prompts import event_extraction_prompt
from stock_screener_engine.llm.base.validators import validate_event
from stock_screener_engine.nlp.schemas.events import EventType, ExpectedDirection, ExtractedEvent, NormalizedDocument


class LLMEventExtractor:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def extract(self, doc: NormalizedDocument, entities: dict[str, list[str]], time_decay_factor: float) -> ExtractedEvent | None:
        payload: dict[str, object] = {"title": doc.title, "text": doc.body_text, "symbol": doc.symbol}
        raw = self.client.complete(task="event_extraction", prompt=event_extraction_prompt(), payload=payload)
        parsed = validate_event(raw)
        if parsed is None:
            return None

        return ExtractedEvent(
            event_type=EventType(parsed.event_type),
            timestamp=doc.timestamp,
            symbol=doc.symbol,
            confidence=parsed.confidence,
            source_type=doc.source,
            summary=parsed.short_summary or doc.title,
            entities={**entities, "keywords": parsed.keywords},
            event_strength=parsed.event_strength,
            expected_direction=ExpectedDirection(parsed.expected_direction),
            time_decay_factor=time_decay_factor,
            uncertainty_score=parsed.uncertainty_score,
            time_horizon=parsed.time_horizon,
            risk_flag=parsed.risk_flag,
            catalyst_flag=parsed.catalyst_flag,
        )
