"""Aggregate document analyses into stable numeric text features with time decay."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from stock_screener_engine.nlp.schemas.events import (
    DocumentAnalysis,
    ExpectedDirection,
    EventType,
    TextFeatureSet,
    TextFeatureVector,
)


class EventFeatureAggregator:
    def __init__(self, half_life_days: float = 14.0, high_impact_threshold: float = 0.65) -> None:
        self.half_life_days = max(1.0, half_life_days)
        self.high_impact_threshold = max(0.0, min(1.0, high_impact_threshold))

    def aggregate(self, analyses: list[DocumentAnalysis], as_of: datetime) -> TextFeatureSet:
        by_symbol: dict[str, list[DocumentAnalysis]] = defaultdict(list)
        for a in analyses:
            by_symbol[a.symbol].append(a)

        vectors: dict[str, TextFeatureVector] = {}
        for symbol, rows in by_symbol.items():
            ordered = sorted(rows, key=lambda r: r.event.timestamp)
            total_decay = 0.0
            pos = 0.0
            neg = 0.0
            weighted = 0.0
            event_strength = 0.0
            event_count = float(len(rows))
            high_impact = 0.0
            uncertainty = 0.0
            governance_hits = 0.0
            catalyst = 0.0
            sentiment_recent = 0.0
            sentiment_trend = 0.0
            sentiment_momentum = 0.0
            management_tone = 0.0
            transcript_quality = 0.0
            earnings_sentiment = 0.0
            catalyst_strength = 0.0
            governance_risk_raw = 0.0
            directions: list[float] = []

            for idx, a in enumerate(ordered):
                decay = _time_decay(days=(as_of - a.event.timestamp).total_seconds() / 86400.0, half_life=self.half_life_days)
                total_decay += decay
                signed = a.event.event_strength * decay
                if a.event.expected_direction == ExpectedDirection.POSITIVE:
                    pos += signed
                    directions.append(1.0)
                elif a.event.expected_direction == ExpectedDirection.NEGATIVE:
                    neg += signed
                    directions.append(-1.0)
                else:
                    directions.append(0.0)

                weighted += signed * (1.0 - a.event.uncertainty_score)
                event_strength += a.event.event_strength * decay
                uncertainty += a.event.uncertainty_score * decay
                if a.event.event_strength >= self.high_impact_threshold:
                    high_impact = 1.0
                if a.event.event_type in {EventType.INSIDER_ACTIVITY, EventType.MANAGEMENT_CHANGE, EventType.REGULATORY_EVENT, EventType.LITIGATION}:
                    governance_hits += 1.0
                    governance_risk_raw += a.event.uncertainty_score * decay
                if a.event.event_type in {EventType.ORDER_WIN, EventType.GUIDANCE_CHANGE, EventType.EARNINGS_RESULT, EventType.CAPEX_ANNOUNCEMENT}:
                    catalyst = 1.0
                    catalyst_strength += a.event.event_strength * decay

                management_tone += a.management_tone_score * decay
                transcript_quality += a.transcript_quality_signal * decay
                earnings_sentiment += a.balance_sheet_sentiment * decay

                sent_values = [s.polarity_score * s.intensity for s in a.sentiments]
                sent_score = sum(sent_values) / len(sent_values) if sent_values else 0.0
                sentiment_recent += sent_score * decay
                if idx > 0:
                    prev = ordered[idx - 1]
                    prev_values = [s.polarity_score * s.intensity for s in prev.sentiments]
                    prev_score = sum(prev_values) / len(prev_values) if prev_values else 0.0
                    sentiment_trend += (sent_score - prev_score)

            denom = max(1e-9, total_decay)
            event_momentum = _normalize_0_1(sum(max(0.0, d) for d in directions) / max(1.0, len(directions)))
            sentiment_recent_norm = max(-1.0, min(1.0, sentiment_recent / denom))
            sentiment_trend_norm = max(-1.0, min(1.0, sentiment_trend))
            sentiment_momentum = max(-1.0, min(1.0, 0.6 * sentiment_trend_norm + 0.4 * sentiment_recent_norm))
            cluster_score = _normalize_0_1(event_count / max(2.0, self.half_life_days / 2.0))
            decayed_event_signal = max(-1.0, min(1.0, (pos - neg) / denom))

            vectors[symbol] = TextFeatureVector(
                symbol=symbol,
                as_of=as_of,
                recent_event_count=_normalize_0_1(event_count / 10.0),
                positive_event_score=_normalize_0_1(pos / denom),
                negative_event_score=_normalize_0_1(neg / denom),
                event_momentum=event_momentum,
                event_decay_weighted_score=max(-1.0, min(1.0, weighted / denom)),
                high_impact_event_flag=high_impact,
                uncertainty_penalty=_normalize_0_1(uncertainty / denom),
                sentiment_score_recent=sentiment_recent_norm,
                sentiment_trend=sentiment_trend_norm,
                sentiment_momentum=sentiment_momentum,
                event_strength_score=_normalize_0_1(event_strength / denom),
                event_risk_score=_normalize_0_1((neg + uncertainty) / denom),
                governance_flag_score=_normalize_0_1(governance_hits / max(1.0, event_count)),
                catalyst_presence_flag=catalyst,
                management_tone_score=max(-1.0, min(1.0, management_tone / denom)),
                earnings_sentiment_score=max(-1.0, min(1.0, earnings_sentiment / denom)),
                event_cluster_score=cluster_score,
                decayed_event_signal=decayed_event_signal,
                transcript_quality_signal=_normalize_0_1(transcript_quality / denom),
                recent_positive_event_score=_normalize_0_1(pos / denom),
                recent_negative_event_score=_normalize_0_1(neg / denom),
                catalyst_strength_score=_normalize_0_1(catalyst_strength / denom),
                governance_risk_score=_normalize_0_1(governance_risk_raw / denom),
            )

        return TextFeatureSet(vectors=vectors)


def _time_decay(days: float, half_life: float) -> float:
    if days <= 0:
        return 1.0
    return 0.5 ** (days / max(1e-9, half_life))


def _normalize_0_1(value: float) -> float:
    return max(0.0, min(1.0, value))
