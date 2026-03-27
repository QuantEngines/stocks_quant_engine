"""Named constants for all feature keys used across the system.

Using string constants instead of bare literals ensures:

* IDE refactoring support
* Compile-time typo detection in tests
* Consistent naming across features, scoring, and explainability modules
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Fundamental features
# ---------------------------------------------------------------------------
FEAT_GROWTH_QUALITY = "growth_quality"
FEAT_PROFITABILITY = "profitability_quality"
FEAT_BALANCE_SHEET_HEALTH = "balance_sheet_health"
FEAT_CASH_FLOW_QUALITY = "cash_flow_quality"
FEAT_VALUATION_SANITY = "valuation_sanity"
FEAT_GOVERNANCE_PROXY = "governance_proxy"
FEAT_REVENUE_GROWTH = "revenue_growth"
FEAT_OPERATING_MARGIN = "operating_margin"
FEAT_EARNINGS_STABILITY = "earnings_stability"
FEAT_LEVERAGE_TREND = "leverage_trend"
FEAT_SECTOR_PE_ZSCORE = "sector_pe_zscore"
FEAT_SECTOR_PB_ZSCORE = "sector_pb_zscore"
FEAT_ROLLING_PE_ZSCORE = "rolling_pe_zscore"
FEAT_ROLLING_PB_ZSCORE = "rolling_pb_zscore"

# ---------------------------------------------------------------------------
# Technical features
# ---------------------------------------------------------------------------
FEAT_TREND_STRENGTH = "trend_strength"
FEAT_MOMENTUM_STRENGTH = "momentum_strength"
FEAT_VOLATILITY_REGIME = "volatility_regime"
FEAT_VOLUME_CONFIRMATION = "volume_confirmation"
FEAT_RELATIVE_STRENGTH = "relative_strength_proxy"
FEAT_DELIVERY_RATIO = "delivery_ratio_signal"

# ---------------------------------------------------------------------------
# Event/governance features
# ---------------------------------------------------------------------------
FEAT_EVENT_CATALYST = "event_catalyst"
FEAT_GOVERNANCE_EVENT = "governance_event_proxy"

# ---------------------------------------------------------------------------
# Sentiment features
# ---------------------------------------------------------------------------
FEAT_SENTIMENT_SCORE = "sentiment_score"
FEAT_NEWS_SENTIMENT = "news_sentiment"

# ---------------------------------------------------------------------------
# Market regime features
# ---------------------------------------------------------------------------
FEAT_MARKET_REGIME = "market_regime_score"
FEAT_SECTOR_MOMENTUM = "sector_momentum"

# ---------------------------------------------------------------------------
# Text / NLP derived features
# ---------------------------------------------------------------------------
FEAT_SENTIMENT_RECENT = "sentiment_score_recent"
FEAT_SENTIMENT_TREND = "sentiment_trend"
FEAT_SENTIMENT_MOMENTUM = "sentiment_momentum"
FEAT_EVENT_STRENGTH_SCORE = "event_strength_score"
FEAT_EVENT_RISK_SCORE = "event_risk_score"
FEAT_GOVERNANCE_FLAG_SCORE = "governance_flag_score"
FEAT_CATALYST_PRESENCE_FLAG = "catalyst_presence_flag"
FEAT_EVENT_MOMENTUM_SCORE = "event_momentum_score"
FEAT_EVENT_DECAY_WEIGHTED_SCORE = "event_decay_weighted_score"
FEAT_UNCERTAINTY_PENALTY = "uncertainty_penalty"
FEAT_RECENT_EVENT_COUNT = "recent_event_count"
FEAT_HIGH_IMPACT_EVENT_FLAG = "high_impact_event_flag"
FEAT_MANAGEMENT_TONE_SCORE = "management_tone_score"
FEAT_EARNINGS_SENTIMENT_SCORE = "earnings_sentiment_score"
FEAT_EVENT_CLUSTER_SCORE = "event_cluster_score"
FEAT_DECAYED_EVENT_SIGNAL = "decayed_event_signal"
FEAT_TRANSCRIPT_QUALITY_SIGNAL = "transcript_quality_signal"
FEAT_RECENT_POSITIVE_EVENT_SCORE = "recent_positive_event_score"
FEAT_RECENT_NEGATIVE_EVENT_SCORE = "recent_negative_event_score"
FEAT_CATALYST_STRENGTH_SCORE = "catalyst_strength_score"
FEAT_GOVERNANCE_RISK_SCORE = "governance_risk_score"

# ---------------------------------------------------------------------------
# Feature group sets (for validation and subsetting)
# ---------------------------------------------------------------------------
FUNDAMENTAL_FEATURES: frozenset[str] = frozenset({
    FEAT_GROWTH_QUALITY,
    FEAT_PROFITABILITY,
    FEAT_BALANCE_SHEET_HEALTH,
    FEAT_CASH_FLOW_QUALITY,
    FEAT_VALUATION_SANITY,
    FEAT_GOVERNANCE_PROXY,
    FEAT_REVENUE_GROWTH,
    FEAT_OPERATING_MARGIN,
    FEAT_EARNINGS_STABILITY,
    FEAT_LEVERAGE_TREND,
    FEAT_SECTOR_PE_ZSCORE,
    FEAT_SECTOR_PB_ZSCORE,
    FEAT_ROLLING_PE_ZSCORE,
    FEAT_ROLLING_PB_ZSCORE,
})

TECHNICAL_FEATURES: frozenset[str] = frozenset({
    FEAT_TREND_STRENGTH,
    FEAT_MOMENTUM_STRENGTH,
    FEAT_VOLATILITY_REGIME,
    FEAT_VOLUME_CONFIRMATION,
    FEAT_RELATIVE_STRENGTH,
    FEAT_DELIVERY_RATIO,
})

EVENT_FEATURES: frozenset[str] = frozenset({
    FEAT_EVENT_CATALYST,
    FEAT_GOVERNANCE_EVENT,
})

SENTIMENT_FEATURES: frozenset[str] = frozenset({
    FEAT_SENTIMENT_SCORE,
    FEAT_NEWS_SENTIMENT,
})

REGIME_FEATURES: frozenset[str] = frozenset({
    FEAT_MARKET_REGIME,
    FEAT_SECTOR_MOMENTUM,
})

TEXT_FEATURES: frozenset[str] = frozenset({
    FEAT_SENTIMENT_RECENT,
    FEAT_SENTIMENT_TREND,
    FEAT_SENTIMENT_MOMENTUM,
    FEAT_EVENT_STRENGTH_SCORE,
    FEAT_EVENT_RISK_SCORE,
    FEAT_GOVERNANCE_FLAG_SCORE,
    FEAT_CATALYST_PRESENCE_FLAG,
    FEAT_EVENT_MOMENTUM_SCORE,
    FEAT_EVENT_DECAY_WEIGHTED_SCORE,
    FEAT_UNCERTAINTY_PENALTY,
    FEAT_RECENT_EVENT_COUNT,
    FEAT_HIGH_IMPACT_EVENT_FLAG,
    FEAT_MANAGEMENT_TONE_SCORE,
    FEAT_EARNINGS_SENTIMENT_SCORE,
    FEAT_EVENT_CLUSTER_SCORE,
    FEAT_DECAYED_EVENT_SIGNAL,
    FEAT_TRANSCRIPT_QUALITY_SIGNAL,
    FEAT_RECENT_POSITIVE_EVENT_SCORE,
    FEAT_RECENT_NEGATIVE_EVENT_SCORE,
    FEAT_CATALYST_STRENGTH_SCORE,
    FEAT_GOVERNANCE_RISK_SCORE,
})

ALL_FEATURES: frozenset[str] = (
    FUNDAMENTAL_FEATURES
    | TECHNICAL_FEATURES
    | EVENT_FEATURES
    | SENTIMENT_FEATURES
    | REGIME_FEATURES
    | TEXT_FEATURES
)
