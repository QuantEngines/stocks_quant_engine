"""LLM-assisted document classification with validated output."""

from __future__ import annotations

from stock_screener_engine.llm.base.llm_client import LLMClient
from stock_screener_engine.llm.base.prompts import classification_prompt
from stock_screener_engine.llm.base.validators import validate_classification
from stock_screener_engine.nlp.schemas.events import DocumentCategory, NormalizedDocument


class LLMDocumentClassifier:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def classify(self, doc: NormalizedDocument) -> tuple[DocumentCategory, float]:
        payload: dict[str, object] = {"title": doc.title, "text": doc.body_text, "symbol": doc.symbol}
        raw = self.client.complete(task="classification", prompt=classification_prompt(), payload=payload)
        parsed = validate_classification(raw)
        if parsed is None:
            return DocumentCategory.UNKNOWN, 0.0
        return DocumentCategory(parsed.category), parsed.confidence
