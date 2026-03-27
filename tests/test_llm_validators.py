from __future__ import annotations

from stock_screener_engine.llm.base.validators import (
    validate_classification,
    validate_event,
    validate_management_tone,
    validate_sentiment,
)


def test_validator_maps_synonyms_and_clips_ranges() -> None:
    cls = validate_classification({"category": "earnings", "confidence": 1.5})
    assert cls is not None
    assert cls.category == "earnings_related"
    assert cls.confidence == 1.0

    event = validate_event(
        {
            "event_type": "litigation_regulatory",
            "expected_direction": "down",
            "event_strength": 1.2,
            "confidence": -1,
            "uncertainty_score": 2,
            "time_horizon": "near_term",
            "risk_flag": True,
            "catalyst_flag": False,
            "keywords": ["penalty"],
        }
    )
    assert event is not None
    assert event.event_type == "litigation"
    assert event.expected_direction == "negative"
    assert event.event_strength == 1.0
    assert event.confidence == 0.0


def test_sentiment_and_management_validation_outputs() -> None:
    s = validate_sentiment({"earnings_sentiment": 2, "guidance_sentiment": -2, "confidence": 0.7})
    assert s.earnings_sentiment == 1.0
    assert s.guidance_sentiment == -1.0
    assert s.confidence == 0.7

    m = validate_management_tone({"management_tone_score": 2.2, "risk_commentary_score": -3})
    assert m.management_tone_score == 1.0
    assert m.risk_commentary_score == -1.0
