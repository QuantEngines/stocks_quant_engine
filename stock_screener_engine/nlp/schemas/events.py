"""Structured schemas for NLP documents, events, and feature outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    NEWS = "news"
    FILING = "filing"
    TRANSCRIPT = "transcript"
    ANNOUNCEMENT = "announcement"
    INTERVIEW = "interview"
    REPORT = "report"


class EventType(str, Enum):
    EARNINGS_RESULT = "earnings_result"
    GUIDANCE_CHANGE = "guidance_change"
    ORDER_WIN = "order_win"
    CAPEX_ANNOUNCEMENT = "capex_announcement"
    MERGER_ACQUISITION = "merger_acquisition"
    MANAGEMENT_CHANGE = "management_change"
    INSIDER_ACTIVITY = "insider_activity"
    REGULATORY_EVENT = "regulatory_event"
    LITIGATION = "litigation"
    CREDIT_RATING_CHANGE = "credit_rating_change"
    MACRO_LINKED_EVENT = "macro_linked_event"
    UNKNOWN = "unknown"


class ExpectedDirection(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class SentimentType(str, Enum):
    EARNINGS = "earnings"
    GOVERNANCE = "governance"
    OUTLOOK = "outlook"
    MACRO = "macro"


class DocumentCategory(str, Enum):
    EARNINGS_RELATED = "earnings_related"
    CORPORATE_ACTION = "corporate_action"
    MANAGEMENT_COMMENTARY = "management_commentary"
    MACRO_SECTOR = "macro_sector"
    GENERAL_NEWS = "general_news"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class NormalizedDocument:
    id: str
    source: SourceType
    timestamp: datetime
    symbol: str
    title: str
    body_text: str
    company_name: str = ""
    source_name: str = ""
    url: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedEvent:
    event_type: EventType
    timestamp: datetime
    symbol: str
    confidence: float
    source_type: SourceType
    summary: str
    entities: dict[str, list[str]]
    event_strength: float
    expected_direction: ExpectedDirection
    time_decay_factor: float
    uncertainty_score: float
    time_horizon: str = "near_term"
    risk_flag: bool = False
    catalyst_flag: bool = False


@dataclass(frozen=True)
class SentimentSignal:
    polarity_score: float
    confidence: float
    sentiment_type: SentimentType
    intensity: float


@dataclass(frozen=True)
class DocumentAnalysis:
    document_id: str
    symbol: str
    category: DocumentCategory
    event: ExtractedEvent
    sentiments: list[SentimentSignal]
    management_tone_score: float = 0.0
    transcript_quality_signal: float = 0.0
    balance_sheet_sentiment: float = 0.0
    llm_confidence: float = 0.0


@dataclass(frozen=True)
class TextFeatureVector:
    symbol: str
    as_of: datetime
    recent_event_count: float
    positive_event_score: float
    negative_event_score: float
    event_momentum: float
    event_decay_weighted_score: float
    high_impact_event_flag: float
    uncertainty_penalty: float
    sentiment_score_recent: float
    sentiment_trend: float
    sentiment_momentum: float
    event_strength_score: float
    event_risk_score: float
    governance_flag_score: float
    catalyst_presence_flag: float
    management_tone_score: float
    earnings_sentiment_score: float
    event_cluster_score: float
    decayed_event_signal: float
    transcript_quality_signal: float
    recent_positive_event_score: float
    recent_negative_event_score: float
    catalyst_strength_score: float
    governance_risk_score: float


@dataclass(frozen=True)
class TextFeatureSet:
    vectors: dict[str, TextFeatureVector]
