"""LLM-assisted management commentary extraction for transcript/interview documents."""

from __future__ import annotations

from stock_screener_engine.llm.base.llm_client import LLMClient
from stock_screener_engine.llm.base.prompts import management_tone_prompt
from stock_screener_engine.llm.base.validators import validate_management_tone
from stock_screener_engine.nlp.schemas.events import NormalizedDocument


class LLMManagementToneExtractor:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def extract(self, doc: NormalizedDocument) -> tuple[float, float, float, float]:
        payload: dict[str, object] = {"title": doc.title, "text": doc.body_text, "symbol": doc.symbol}
        raw = self.client.complete(task="management_tone", prompt=management_tone_prompt(), payload=payload)
        parsed = validate_management_tone(raw)

        tone = parsed.management_tone_score
        quality = 0.4 * abs(parsed.margin_commentary_score) + 0.3 * abs(parsed.demand_commentary_score) + 0.3 * abs(parsed.capital_allocation_stance)
        risk = max(0.0, min(1.0, (parsed.risk_commentary_score + 1.0) / 2.0))
        return tone, quality * parsed.confidence, risk, parsed.confidence
