"""Conviction scoring for cross-horizon, risk-aware signal confidence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from stock_screener_engine.core.normalizers import clamp


_FUNDAMENTAL_KEYS = (
    "growth_quality",
    "profitability_quality",
    "balance_sheet_health",
    "cash_flow_quality",
    "revenue_growth",
    "operating_margin",
    "earnings_stability",
)
_VALUATION_KEYS = (
    "valuation_sanity",
    "pe_ratio",
    "pb_ratio",
    "debt_to_equity",
    "cfo_pat_ratio",
)
_TECHNICAL_KEYS = (
    "trend_strength",
    "momentum_strength",
    "relative_strength_proxy",
    "volatility_regime",
    "volume_confirmation",
    "activity_vs_avg",
    "breakout_score",
    "compression_score",
)
_EVENT_NLP_KEYS = (
    "event_catalyst",
    "sentiment_score",
    "sentiment_trend",
    "management_tone_score",
    "catalyst_strength_score",
    "uncertainty_penalty",
    "transcript_quality_signal",
)
_REGIME_KEYS = (
    "market_regime_score",
    "sector_momentum",
    "cross_sectional_momentum_rank",
    "sector_relative_momentum_rank",
)

_SOURCE_CONFIDENCE_KEYS = (
    "source_confidence",
    "data_source_confidence",
    "market_data_confidence",
    "fundamental_data_confidence",
    "financials_data_confidence",
    "banking_data_confidence",
    "document_quality_score",
    "document_extraction_quality",
    "source_reconciliation_score",
    "llm_confidence",
)
_BACKTEST_EVIDENCE_KEYS = (
    "backtest_hit_rate",
    "backtest_precision",
    "backtest_win_rate",
    "backtest_information_coefficient",
    "factor_ic",
    "ranking_ic",
    "walk_forward_score",
    "calibration_score",
)


@dataclass(frozen=True)
class ConvictionWeights:
    signal_agreement: float = 0.20
    data_completeness: float = 0.20
    source_confidence: float = 0.10
    backtest_evidence: float = 0.10
    sector_regime_confirmation: float = 0.15
    risk_resilience: float = 0.25

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "ConvictionWeights":
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ConvictionResult:
    score: float
    components: dict[str, float]


@dataclass
class ConvictionScorer:
    """Convert cross-horizon scores into an auditable conviction score.

    The old conviction score is retained as ``score_strength``. The final score
    discounts that base by support evidence, so poor data coverage or weak
    signal agreement cannot create conviction when the underlying score is low.
    """

    weights: ConvictionWeights = field(default_factory=ConvictionWeights)
    max_risk_penalty: float = 30.0
    min_support_multiplier: float = 0.55

    def score(
        self,
        features: Mapping[str, float] | None,
        adjusted_long_score: float,
        adjusted_swing_score: float,
        risk_penalty: float,
    ) -> ConvictionResult:
        values = features or {}
        score_strength = _clamp_0_100((adjusted_long_score + adjusted_swing_score) / 2.0)
        signal_agreement = self._signal_agreement(adjusted_long_score, adjusted_swing_score, score_strength)
        data_completeness = self._data_completeness(values)
        source_confidence = self._source_confidence(values)
        backtest_evidence = self._backtest_evidence(values)
        sector_regime_confirmation = self._sector_regime_confirmation(values)
        risk_resilience = self._risk_resilience(risk_penalty)

        support_score = self._weighted_support(
            {
                "signal_agreement": signal_agreement,
                "data_completeness": data_completeness,
                "source_confidence": source_confidence,
                "backtest_evidence": backtest_evidence,
                "sector_regime_confirmation": sector_regime_confirmation,
                "risk_resilience": risk_resilience,
            }
        )
        multiplier = self.min_support_multiplier + (1.0 - self.min_support_multiplier) * (support_score / 100.0)
        conviction = _clamp_0_100(score_strength * multiplier)

        return ConvictionResult(
            score=conviction,
            components={
                "score_strength": round(score_strength, 6),
                "signal_agreement": round(signal_agreement, 6),
                "data_completeness": round(data_completeness, 6),
                "source_confidence": round(source_confidence, 6),
                "backtest_evidence": round(backtest_evidence, 6),
                "sector_regime_confirmation": round(sector_regime_confirmation, 6),
                "risk_resilience": round(risk_resilience, 6),
                "support_score": round(support_score, 6),
                "support_multiplier": round(multiplier, 6),
            },
        )

    def _signal_agreement(self, long_score: float, swing_score: float, score_strength: float) -> float:
        distance_penalty = clamp(abs(long_score - swing_score) / 100.0)
        agreement = 1.0 - distance_penalty
        if score_strength < 20.0:
            agreement *= score_strength / 20.0
        return _clamp_0_100(agreement * 100.0)

    def _data_completeness(self, values: Mapping[str, float]) -> float:
        fundamental = _bucket_coverage(values, _FUNDAMENTAL_KEYS, all_zero_floor=0.0)
        valuation = _bucket_coverage(values, _VALUATION_KEYS, all_zero_floor=0.0)
        technical = _bucket_coverage(values, _TECHNICAL_KEYS, all_zero_floor=0.7)
        event_nlp = _bucket_coverage(values, _EVENT_NLP_KEYS, all_zero_floor=0.5)
        regime = _bucket_coverage(values, _REGIME_KEYS, all_zero_floor=0.8)
        completeness = (
            0.30 * fundamental
            + 0.20 * valuation
            + 0.25 * technical
            + 0.15 * event_nlp
            + 0.10 * regime
        )
        explicit_coverage = _bounded_feature(values, "feature_coverage_score")
        if explicit_coverage is not None:
            completeness = 0.70 * completeness + 0.30 * explicit_coverage
        return _clamp_0_100(completeness * 100.0)

    def _source_confidence(self, values: Mapping[str, float]) -> float:
        explicit = [_bounded_feature(values, key) for key in _SOURCE_CONFIDENCE_KEYS if key in values]
        explicit = [x for x in explicit if x is not None]
        if not explicit:
            return 50.0
        return _clamp_0_100((sum(explicit) / len(explicit)) * 100.0)

    def _backtest_evidence(self, values: Mapping[str, float]) -> float:
        evidence: list[float] = []
        for key in _BACKTEST_EVIDENCE_KEYS:
            if key not in values:
                continue
            value = _finite_float(values.get(key))
            if value is None:
                continue
            if key.endswith("_ic") or "information_coefficient" in key:
                evidence.append(clamp((value + 0.05) / 0.15))
            else:
                evidence.append(clamp(value / 100.0 if value > 1.0 else value))
        if not evidence:
            return 50.0
        return _clamp_0_100((sum(evidence) / len(evidence)) * 100.0)

    def _sector_regime_confirmation(self, values: Mapping[str, float]) -> float:
        regime = _signed_feature_to_confirmation(values, "market_regime_score")
        sector = _signed_feature_to_confirmation(values, "sector_momentum")
        universe_rank = _bounded_feature(values, "cross_sectional_momentum_rank")
        sector_rank = _bounded_feature(values, "sector_relative_momentum_rank")
        universe_rank = 0.5 if universe_rank is None else universe_rank
        sector_rank = 0.5 if sector_rank is None else sector_rank
        return _clamp_0_100(
            (0.25 * regime + 0.30 * sector + 0.20 * universe_rank + 0.25 * sector_rank) * 100.0
        )

    def _risk_resilience(self, risk_penalty: float) -> float:
        max_penalty = max(1e-9, float(self.max_risk_penalty))
        return _clamp_0_100((1.0 - clamp(risk_penalty / max_penalty)) * 100.0)

    def _weighted_support(self, components: Mapping[str, float]) -> float:
        total_weight = sum(max(0.0, getattr(self.weights, name)) for name in components)
        if total_weight <= 0:
            return 50.0
        weighted = sum(max(0.0, getattr(self.weights, name)) * value for name, value in components.items())
        return _clamp_0_100(weighted / total_weight)


def _bucket_coverage(values: Mapping[str, float], keys: tuple[str, ...], all_zero_floor: float) -> float:
    present_values: list[float] = []
    for key in keys:
        if key not in values:
            continue
        value = _finite_float(values.get(key))
        if value is not None:
            present_values.append(value)
    if not keys:
        return 1.0
    coverage = len(present_values) / len(keys)
    if present_values and all(abs(value) <= 1e-12 for value in present_values):
        coverage *= clamp(all_zero_floor)
    return clamp(coverage)


def _bounded_feature(values: Mapping[str, float], key: str) -> float | None:
    value = _finite_float(values.get(key))
    if value is None:
        return None
    return clamp(value / 100.0 if value > 1.0 else value)


def _signed_feature_to_confirmation(values: Mapping[str, float], key: str) -> float:
    value = _finite_float(values.get(key))
    if value is None:
        return 0.5
    return clamp((value + 1.0) / 2.0)


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _clamp_0_100(value: float) -> float:
    return max(0.0, min(100.0, float(value)))
