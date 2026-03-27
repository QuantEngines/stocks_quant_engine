"""Swing-trade scoring engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from stock_screener_engine.core.feature_access import FeatureAccessor
from stock_screener_engine.core.normalizers import clamp, linear_score
from stock_screener_engine.core.scoring_base import CategoryScore, ScoringResult, combine_categories


@dataclass(frozen=True)
class SwingCategoryWeights:
    trend: float = 0.22
    momentum: float = 0.20
    volume_participation: float = 0.18
    volatility_regime: float = 0.14
    setup_quality: float = 0.16
    catalyst_awareness: float = 0.10


@dataclass
class SwingTradeScorer:
    weights: SwingCategoryWeights = field(default_factory=SwingCategoryWeights)

    def score(self, features: Mapping[str, float] | None) -> ScoringResult:
        fa = FeatureAccessor(features)
        categories = [
            self._trend(fa),
            self._momentum(fa),
            self._volume(fa),
            self._volatility(fa),
            self._setup_quality(fa),
            self._catalyst_awareness(fa),
        ]
        result = combine_categories(categories, scale=100.0)
        return ScoringResult(
            total_score=result.total_score,
            categories=result.categories,
            component_map=result.component_map,
            missing_features=sorted(set(result.missing_features + fa.missing_features() + fa.invalid_features())),
        )

    def _trend(self, fa: FeatureAccessor) -> CategoryScore:
        trend = clamp(fa.get("trend_strength", 0.0))
        rel = clamp(fa.get("relative_strength_proxy", 0.0))
        score = clamp(0.7 * trend + 0.3 * rel)
        return CategoryScore(
            name="trend",
            score_0_1=score,
            weight=self.weights.trend,
            contribution=score * self.weights.trend,
            missing_features=[k for k in ["trend_strength", "relative_strength_proxy"] if k in fa.missing],
        )

    def _momentum(self, fa: FeatureAccessor) -> CategoryScore:
        mom = clamp(fa.get("momentum_strength", 0.0))
        accel = linear_score(fa.get("price_acceleration", 0.0), -0.05, 0.10)
        score = clamp(0.8 * mom + 0.2 * accel)
        return CategoryScore(
            name="momentum",
            score_0_1=score,
            weight=self.weights.momentum,
            contribution=score * self.weights.momentum,
            missing_features=[k for k in ["momentum_strength", "price_acceleration"] if k in fa.missing],
        )

    def _volume(self, fa: FeatureAccessor) -> CategoryScore:
        vol = clamp(fa.get("volume_confirmation", 0.0))
        activity = linear_score(fa.get("activity_vs_avg", 1.0), 0.8, 2.0)
        score = clamp(0.75 * vol + 0.25 * activity)
        return CategoryScore(
            name="volume_participation",
            score_0_1=score,
            weight=self.weights.volume_participation,
            contribution=score * self.weights.volume_participation,
            missing_features=[k for k in ["volume_confirmation", "activity_vs_avg"] if k in fa.missing],
        )

    def _volatility(self, fa: FeatureAccessor) -> CategoryScore:
        vol_regime = clamp(fa.get("volatility_regime", 0.0))
        score = vol_regime
        return CategoryScore(
            name="volatility_regime",
            score_0_1=score,
            weight=self.weights.volatility_regime,
            contribution=score * self.weights.volatility_regime,
            missing_features=[k for k in ["volatility_regime"] if k in fa.missing],
        )

    def _setup_quality(self, fa: FeatureAccessor) -> CategoryScore:
        trend = clamp(fa.get("trend_strength", 0.0))
        compression = linear_score(fa.get("compression_score", 0.0), 0.2, 0.9)
        breakout = linear_score(fa.get("breakout_score", 0.0), 0.2, 0.9)
        score = clamp(0.4 * trend + 0.3 * compression + 0.3 * breakout)
        return CategoryScore(
            name="setup_quality",
            score_0_1=score,
            weight=self.weights.setup_quality,
            contribution=score * self.weights.setup_quality,
            missing_features=[k for k in ["trend_strength", "compression_score", "breakout_score"] if k in fa.missing],
        )

    def _catalyst_awareness(self, fa: FeatureAccessor) -> CategoryScore:
        event = fa.get("event_catalyst", 0.0)
        sentiment = fa.get("sentiment_score", 0.0)
        sentiment_momentum = fa.get("sentiment_momentum", 0.0)
        catalyst_flag = clamp(fa.get("catalyst_presence_flag", 0.0))
        event_momentum = clamp(fa.get("event_momentum_score", 0.0))
        catalyst_strength = clamp(fa.get("catalyst_strength_score", 0.0))
        decayed_signal = clamp((fa.get("decayed_event_signal", 0.0) + 1.0) / 2.0)
        base = (0.45 * event + 0.25 * sentiment + 0.15 * sentiment_momentum + 0.15 * (2.0 * decayed_signal - 1.0) + 1.0) / 2.0
        score = clamp(0.65 * base + 0.15 * catalyst_flag + 0.10 * event_momentum + 0.10 * catalyst_strength)
        return CategoryScore(
            name="catalyst_awareness",
            score_0_1=score,
            weight=self.weights.catalyst_awareness,
            contribution=score * self.weights.catalyst_awareness,
            missing_features=[k for k in ["event_catalyst", "sentiment_score", "sentiment_momentum", "catalyst_presence_flag", "event_momentum_score", "catalyst_strength_score", "decayed_event_signal"] if k in fa.missing],
        )
