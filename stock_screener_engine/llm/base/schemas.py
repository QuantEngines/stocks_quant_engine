"""Internal LLM extraction schemas before normalization into NLP canonical schema."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMClassificationOutput:
    category: str
    confidence: float


@dataclass(frozen=True)
class LLMEventOutput:
    event_type: str
    short_summary: str
    expected_direction: str
    event_strength: float
    confidence: float
    uncertainty_score: float
    time_horizon: str
    risk_flag: bool
    catalyst_flag: bool
    keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LLMSentimentOutput:
    earnings_sentiment: float
    guidance_sentiment: float
    governance_sentiment: float
    balance_sheet_sentiment: float
    business_momentum_sentiment: float
    confidence: float


@dataclass(frozen=True)
class LLMManagementToneOutput:
    management_tone_score: float
    expansion_intent: float
    capital_allocation_stance: float
    margin_commentary_score: float
    demand_commentary_score: float
    risk_commentary_score: float
    confidence: float
