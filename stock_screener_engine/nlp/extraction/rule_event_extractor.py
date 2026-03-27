"""Heuristic event extraction into structured and explainable event records."""

from __future__ import annotations

from stock_screener_engine.nlp.schemas.events import (
    EventType,
    ExpectedDirection,
    ExtractedEvent,
    NormalizedDocument,
)

_EVENT_RULES: list[tuple[EventType, tuple[str, ...]]] = [
    (EventType.ORDER_WIN, ("order", "contract", "deal win", "wins")),
    (EventType.GUIDANCE_CHANGE, ("guidance", "outlook raised", "outlook cut")),
    (EventType.EARNINGS_RESULT, ("earnings", "quarterly results", "profit", "margin")),
    (EventType.CAPEX_ANNOUNCEMENT, ("capex", "capacity expansion", "new plant")),
    (EventType.MERGER_ACQUISITION, ("acquisition", "merger", "takeover")),
    (EventType.MANAGEMENT_CHANGE, ("ceo", "cfo", "management change", "resigned")),
    (EventType.INSIDER_ACTIVITY, ("insider", "promoter", "stake purchase", "stake sale")),
    (EventType.REGULATORY_EVENT, ("regulatory", "sebi", "rbi", "fda", "observation")),
    (EventType.LITIGATION, ("litigation", "lawsuit", "penalty", "court")),
    (EventType.CREDIT_RATING_CHANGE, ("credit rating", "downgrade", "upgrade")),
    (EventType.MACRO_LINKED_EVENT, ("inflation", "policy rate", "macro", "sector demand")),
]

_POSITIVE_TERMS = ("win", "upgrade", "raised", "resolved", "strong", "beat", "growth")
_NEGATIVE_TERMS = ("downgrade", "cut", "delay", "miss", "warning", "penalty", "litigation", "concern")


class RuleEventExtractor:
    def extract(
        self,
        doc: NormalizedDocument,
        time_decay_factor: float,
        entities: dict[str, list[str]],
    ) -> ExtractedEvent:
        text = f"{doc.title} {doc.body_text}".lower()
        event_type = EventType.UNKNOWN
        for candidate, keywords in _EVENT_RULES:
            if any(k in text for k in keywords):
                event_type = candidate
                break

        direction = _expected_direction(text)
        strength = _event_strength(text)
        uncertainty = _uncertainty_score(text)
        confidence = max(0.15, min(0.95, 0.35 + 0.4 * strength + 0.25 * (1.0 - uncertainty)))

        return ExtractedEvent(
            event_type=event_type,
            timestamp=doc.timestamp,
            symbol=doc.symbol,
            confidence=confidence,
            source_type=doc.source,
            summary=doc.title,
            entities=entities,
            event_strength=strength,
            expected_direction=direction,
            time_decay_factor=time_decay_factor,
            uncertainty_score=uncertainty,
            time_horizon="near_term",
            risk_flag=direction == ExpectedDirection.NEGATIVE,
            catalyst_flag=direction == ExpectedDirection.POSITIVE,
        )


def _expected_direction(text: str) -> ExpectedDirection:
    pos = sum(t in text for t in _POSITIVE_TERMS)
    neg = sum(t in text for t in _NEGATIVE_TERMS)
    if pos > neg:
        return ExpectedDirection.POSITIVE
    if neg > pos:
        return ExpectedDirection.NEGATIVE
    return ExpectedDirection.NEUTRAL


def _event_strength(text: str) -> float:
    score = 0.2
    if any(token in text for token in ("order", "contract", "guidance", "earnings", "acquisition")):
        score += 0.3
    if any(token in text for token in ("crore", "billion", "%", "margin")):
        score += 0.2
    if any(token in text for token in ("major", "record", "largest")):
        score += 0.2
    return max(0.0, min(1.0, score))


def _uncertainty_score(text: str) -> float:
    tokens = ("may", "could", "uncertain", "subject to", "pending", "possibly")
    score = 0.1 + 0.15 * sum(t in text for t in tokens)
    return max(0.0, min(1.0, score))
