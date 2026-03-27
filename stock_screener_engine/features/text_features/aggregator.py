"""Conversion helpers from NLP feature schemas to plain numeric maps."""

from __future__ import annotations

from stock_screener_engine.nlp.schemas.events import TextFeatureSet


def to_feature_map(text_features: TextFeatureSet) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for symbol, row in text_features.vectors.items():
        out[symbol] = {
            "sentiment_score_recent": row.sentiment_score_recent,
            "sentiment_trend": row.sentiment_trend,
            "sentiment_momentum": row.sentiment_momentum,
            "event_strength_score": row.event_strength_score,
            "event_risk_score": row.event_risk_score,
            "governance_flag_score": row.governance_flag_score,
            "catalyst_presence_flag": row.catalyst_presence_flag,
            "event_momentum_score": row.event_momentum,
            "event_decay_weighted_score": row.event_decay_weighted_score,
            "uncertainty_penalty": row.uncertainty_penalty,
            "recent_event_count": row.recent_event_count,
            "high_impact_event_flag": row.high_impact_event_flag,
            "management_tone_score": row.management_tone_score,
            "earnings_sentiment_score": row.earnings_sentiment_score,
            "event_cluster_score": row.event_cluster_score,
            "decayed_event_signal": row.decayed_event_signal,
            "transcript_quality_signal": row.transcript_quality_signal,
            "recent_positive_event_score": row.recent_positive_event_score,
            "recent_negative_event_score": row.recent_negative_event_score,
            "catalyst_strength_score": row.catalyst_strength_score,
            "governance_risk_score": row.governance_risk_score,
        }
    return out
