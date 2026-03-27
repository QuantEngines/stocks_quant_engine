"""LLM-assisted finance sentiment extraction mapped to canonical sentiment signals."""

from __future__ import annotations

from stock_screener_engine.llm.base.llm_client import LLMClient
from stock_screener_engine.llm.base.prompts import sentiment_prompt
from stock_screener_engine.llm.base.validators import validate_sentiment
from stock_screener_engine.nlp.schemas.events import SentimentSignal, SentimentType, NormalizedDocument


class LLMSentimentExtractor:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def extract(self, doc: NormalizedDocument) -> tuple[list[SentimentSignal], float]:
        payload: dict[str, object] = {"title": doc.title, "text": doc.body_text, "symbol": doc.symbol}
        raw = self.client.complete(task="sentiment", prompt=sentiment_prompt(), payload=payload)
        parsed = validate_sentiment(raw)

        signals = [
            SentimentSignal(parsed.earnings_sentiment, parsed.confidence, SentimentType.EARNINGS, abs(parsed.earnings_sentiment)),
            SentimentSignal(parsed.governance_sentiment, parsed.confidence, SentimentType.GOVERNANCE, abs(parsed.governance_sentiment)),
            SentimentSignal(parsed.guidance_sentiment, parsed.confidence, SentimentType.OUTLOOK, abs(parsed.guidance_sentiment)),
            SentimentSignal(parsed.business_momentum_sentiment, parsed.confidence, SentimentType.MACRO, abs(parsed.business_momentum_sentiment)),
        ]
        return signals, parsed.balance_sheet_sentiment
