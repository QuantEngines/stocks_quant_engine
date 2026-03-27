"""Reusable internal prompts for structured extraction tasks."""

from __future__ import annotations


def classification_prompt() -> str:
    return (
        "Classify the document into one canonical category. "
        "Return strict JSON only with keys: category, confidence. "
        "Do not invent facts."
    )


def event_extraction_prompt() -> str:
    return (
        "Extract one primary structured event from the document. "
        "Return strict JSON with event_type, short_summary, expected_direction, "
        "event_strength, confidence, uncertainty_score, time_horizon, risk_flag, catalyst_flag, keywords. "
        "If ambiguous, use mixed/neutral semantics and raise uncertainty."
    )


def sentiment_prompt() -> str:
    return (
        "Extract finance-aware sentiment dimensions. "
        "Return strict JSON with earnings_sentiment, guidance_sentiment, governance_sentiment, "
        "balance_sheet_sentiment, business_momentum_sentiment, confidence."
    )


def management_tone_prompt() -> str:
    return (
        "For management commentary/transcript text, return strict JSON with "
        "management_tone_score, expansion_intent, capital_allocation_stance, "
        "margin_commentary_score, demand_commentary_score, risk_commentary_score, confidence."
    )
