"""Validation and normalization for LLM extraction payloads."""

from __future__ import annotations

from stock_screener_engine.llm.base.schemas import (
    LLMClassificationOutput,
    LLMEventOutput,
    LLMManagementToneOutput,
    LLMSentimentOutput,
)
from stock_screener_engine.nlp.schemas.events import DocumentCategory, EventType, ExpectedDirection

_EVENT_SYNONYMS = {
    "litigation_regulatory": "litigation",
    "regulatory": "regulatory_event",
    "ma": "merger_acquisition",
    "m&a": "merger_acquisition",
    "insider": "insider_activity",
}

_CATEGORY_SYNONYMS = {
    "earnings": "earnings_related",
    "corporate": "corporate_action",
    "management": "management_commentary",
    "macro": "macro_sector",
}

_DIRECTION_SYNONYMS = {
    "up": "positive",
    "down": "negative",
    "mixed": "neutral",
}


def validate_classification(payload: dict[str, object]) -> LLMClassificationOutput | None:
    category_raw = str(payload.get("category", "unknown")).strip().lower()
    category_raw = _CATEGORY_SYNONYMS.get(category_raw, category_raw)
    category = category_raw if category_raw in {c.value for c in DocumentCategory} else DocumentCategory.UNKNOWN.value
    confidence = _clip(payload.get("confidence", 0.0))
    return LLMClassificationOutput(category=category, confidence=confidence)


def validate_event(payload: dict[str, object]) -> LLMEventOutput | None:
    event_raw = str(payload.get("event_type", "unknown")).strip().lower()
    event_raw = _EVENT_SYNONYMS.get(event_raw, event_raw)
    event_type = event_raw if event_raw in {e.value for e in EventType} else EventType.UNKNOWN.value

    direction_raw = str(payload.get("expected_direction", "neutral")).strip().lower()
    direction_raw = _DIRECTION_SYNONYMS.get(direction_raw, direction_raw)
    expected_direction = direction_raw if direction_raw in {d.value for d in ExpectedDirection} else ExpectedDirection.NEUTRAL.value

    keywords = payload.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []

    return LLMEventOutput(
        event_type=event_type,
        short_summary=str(payload.get("short_summary", "")).strip()[:240],
        expected_direction=expected_direction,
        event_strength=_clip(payload.get("event_strength", 0.0)),
        confidence=_clip(payload.get("confidence", 0.0)),
        uncertainty_score=_clip(payload.get("uncertainty_score", 0.0)),
        time_horizon=str(payload.get("time_horizon", "near_term")),
        risk_flag=bool(payload.get("risk_flag", False)),
        catalyst_flag=bool(payload.get("catalyst_flag", False)),
        keywords=[str(k) for k in keywords[:20]],
    )


def validate_sentiment(payload: dict[str, object]) -> LLMSentimentOutput:
    return LLMSentimentOutput(
        earnings_sentiment=_clip_sym(payload.get("earnings_sentiment", 0.0)),
        guidance_sentiment=_clip_sym(payload.get("guidance_sentiment", 0.0)),
        governance_sentiment=_clip_sym(payload.get("governance_sentiment", 0.0)),
        balance_sheet_sentiment=_clip_sym(payload.get("balance_sheet_sentiment", 0.0)),
        business_momentum_sentiment=_clip_sym(payload.get("business_momentum_sentiment", 0.0)),
        confidence=_clip(payload.get("confidence", 0.0)),
    )


def validate_management_tone(payload: dict[str, object]) -> LLMManagementToneOutput:
    return LLMManagementToneOutput(
        management_tone_score=_clip_sym(payload.get("management_tone_score", 0.0)),
        expansion_intent=_clip_sym(payload.get("expansion_intent", 0.0)),
        capital_allocation_stance=_clip_sym(payload.get("capital_allocation_stance", 0.0)),
        margin_commentary_score=_clip_sym(payload.get("margin_commentary_score", 0.0)),
        demand_commentary_score=_clip_sym(payload.get("demand_commentary_score", 0.0)),
        risk_commentary_score=_clip_sym(payload.get("risk_commentary_score", 0.0)),
        confidence=_clip(payload.get("confidence", 0.0)),
    )


def _clip(value: object) -> float:
    try:
        return max(0.0, min(1.0, _to_float(value)))
    except (TypeError, ValueError):
        return 0.0


def _clip_sym(value: object) -> float:
    try:
        return max(-1.0, min(1.0, _to_float(value)))
    except (TypeError, ValueError):
        return 0.0


def _to_float(value: object) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError("Unsupported numeric value type")
