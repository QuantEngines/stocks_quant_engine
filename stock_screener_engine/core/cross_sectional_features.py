"""Cross-sectional feature enrichment for signal-first research workflows."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from stock_screener_engine.core.entities import FeatureVector
from stock_screener_engine.core.feature_specs import (
    FEAT_CROSS_SECTIONAL_MOMENTUM_RANK,
    FEAT_CROSS_SECTIONAL_QUALITY_RANK,
    FEAT_CROSS_SECTIONAL_VALUE_RANK,
    FEAT_FEATURE_COVERAGE_SCORE,
    FEAT_LIQUIDITY_PERCENTILE,
    FEAT_QUALITY_VALUE_COMPOSITE,
    FEAT_RESEARCH_READINESS_SCORE,
    FEAT_SECTOR_FEATURE_COVERAGE_SCORE,
    FEAT_SECTOR_RELATIVE_MOMENTUM_RANK,
    FEAT_SECTOR_RELATIVE_QUALITY_RANK,
)


@dataclass(frozen=True)
class CrossSectionalFeatureEnricher:
    """Add universe/sector context without touching IO or scoring state.

    Institutional ranking systems rarely judge a stock in isolation. This
    enricher converts the raw feature vector into peer-aware features that can
    be consumed by scoring, conviction, reports, and backtests.
    """

    minimum_sector_size_for_rank: int = 2

    def enrich(
        self,
        features: Iterable[FeatureVector],
        sector_by_symbol: Mapping[str, str],
    ) -> list[FeatureVector]:
        rows = list(features)
        if not rows:
            return []

        momentum_scores = {fv.symbol: _momentum_composite(fv.values) for fv in rows}
        quality_scores = {fv.symbol: _quality_composite(fv.values) for fv in rows}
        value_scores = {fv.symbol: _value_composite(fv.values) for fv in rows}
        liquidity_scores = {fv.symbol: _liquidity_score(fv.values) for fv in rows}
        coverage_scores = {fv.symbol: _feature_coverage(fv.values) for fv in rows}

        universe_momentum_rank = _percentile_ranks(momentum_scores)
        universe_quality_rank = _percentile_ranks(quality_scores)
        universe_value_rank = _percentile_ranks(value_scores)
        liquidity_rank = _percentile_ranks(liquidity_scores)

        sector_momentum_rank = self._sector_percentile_ranks(momentum_scores, sector_by_symbol)
        sector_quality_rank = self._sector_percentile_ranks(quality_scores, sector_by_symbol)
        sector_coverage = self._sector_averages(coverage_scores, sector_by_symbol)

        enriched: list[FeatureVector] = []
        for fv in rows:
            values = dict(fv.values)
            sector = _sector_key(sector_by_symbol.get(fv.symbol))
            quality_value = _clamp01(
                0.55 * universe_quality_rank.get(fv.symbol, 0.5)
                + 0.45 * universe_value_rank.get(fv.symbol, 0.5)
            )
            readiness = _clamp01(
                0.35 * coverage_scores.get(fv.symbol, 0.0)
                + 0.20 * liquidity_rank.get(fv.symbol, 0.5)
                + 0.20 * quality_value
                + 0.15 * universe_momentum_rank.get(fv.symbol, 0.5)
                + 0.10 * sector_momentum_rank.get(fv.symbol, 0.5)
            )
            values.update(
                {
                    FEAT_CROSS_SECTIONAL_MOMENTUM_RANK: universe_momentum_rank.get(fv.symbol, 0.5),
                    FEAT_SECTOR_RELATIVE_MOMENTUM_RANK: sector_momentum_rank.get(fv.symbol, 0.5),
                    FEAT_CROSS_SECTIONAL_QUALITY_RANK: universe_quality_rank.get(fv.symbol, 0.5),
                    FEAT_SECTOR_RELATIVE_QUALITY_RANK: sector_quality_rank.get(fv.symbol, 0.5),
                    FEAT_CROSS_SECTIONAL_VALUE_RANK: universe_value_rank.get(fv.symbol, 0.5),
                    FEAT_QUALITY_VALUE_COMPOSITE: quality_value,
                    FEAT_LIQUIDITY_PERCENTILE: liquidity_rank.get(fv.symbol, 0.5),
                    FEAT_FEATURE_COVERAGE_SCORE: coverage_scores.get(fv.symbol, 0.0),
                    FEAT_SECTOR_FEATURE_COVERAGE_SCORE: sector_coverage.get(sector, 0.0),
                    FEAT_RESEARCH_READINESS_SCORE: readiness,
                }
            )
            enriched.append(FeatureVector(symbol=fv.symbol, as_of=fv.as_of, values=values))
        return enriched

    def _sector_percentile_ranks(
        self,
        scores: Mapping[str, float],
        sector_by_symbol: Mapping[str, str],
    ) -> dict[str, float]:
        by_sector: dict[str, dict[str, float]] = defaultdict(dict)
        for symbol, score in scores.items():
            by_sector[_sector_key(sector_by_symbol.get(symbol))][symbol] = score

        out: dict[str, float] = {}
        for members in by_sector.values():
            if len(members) < self.minimum_sector_size_for_rank:
                out.update({symbol: 0.5 for symbol in members})
                continue
            out.update(_percentile_ranks(members))
        return out

    def _sector_averages(
        self,
        scores: Mapping[str, float],
        sector_by_symbol: Mapping[str, str],
    ) -> dict[str, float]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for symbol, score in scores.items():
            buckets[_sector_key(sector_by_symbol.get(symbol))].append(score)
        return {
            sector: sum(values) / len(values)
            for sector, values in buckets.items()
            if values
        }


def _momentum_composite(values: Mapping[str, float]) -> float:
    acceleration = _linear(_get(values, "price_acceleration"), -0.05, 0.10)
    return _clamp01(
        0.30 * _get(values, "trend_strength")
        + 0.30 * _get(values, "momentum_strength")
        + 0.20 * _get(values, "relative_strength_proxy")
        + 0.10 * _get(values, "breakout_score", 0.5)
        + 0.10 * acceleration
    )


def _quality_composite(values: Mapping[str, float]) -> float:
    return _clamp01(
        0.24 * _get(values, "growth_quality")
        + 0.22 * _get(values, "profitability_quality")
        + 0.18 * _get(values, "cash_flow_quality")
        + 0.16 * _get(values, "balance_sheet_health")
        + 0.12 * _get(values, "earnings_stability")
        + 0.08 * _get(values, "governance_proxy")
    )


def _value_composite(values: Mapping[str, float]) -> float:
    sector_pe = _inverse_z(_get(values, "sector_pe_zscore"))
    sector_pb = _inverse_z(_get(values, "sector_pb_zscore"))
    rolling_pe = _inverse_z(_get(values, "rolling_pe_zscore"))
    return _clamp01(
        0.55 * _get(values, "valuation_sanity")
        + 0.20 * sector_pe
        + 0.15 * sector_pb
        + 0.10 * rolling_pe
    )


def _liquidity_score(values: Mapping[str, float]) -> float:
    activity = _linear(_get(values, "activity_vs_avg", 1.0), 0.5, 2.0)
    return _clamp01(0.70 * _get(values, "volume_confirmation") + 0.30 * activity)


def _feature_coverage(values: Mapping[str, float]) -> float:
    fundamental = _non_zero_bucket(
        values,
        (
            "growth_quality",
            "profitability_quality",
            "balance_sheet_health",
            "cash_flow_quality",
            "revenue_growth",
            "operating_margin",
        ),
    )
    valuation = _non_zero_bucket(values, ("valuation_sanity", "pe_ratio", "pb_ratio", "cfo_pat_ratio"))
    technical = _finite_bucket(
        values,
        (
            "trend_strength",
            "momentum_strength",
            "relative_strength_proxy",
            "volatility_regime",
            "volume_confirmation",
            "breakout_score",
            "compression_score",
        ),
    )
    event_nlp = _non_zero_bucket(
        values,
        (
            "event_catalyst",
            "sentiment_score",
            "sentiment_trend",
            "management_tone_score",
            "catalyst_strength_score",
            "uncertainty_penalty",
        ),
    )
    regime = _finite_bucket(values, ("market_regime_score", "sector_momentum"))
    return _clamp01(
        0.30 * fundamental
        + 0.20 * valuation
        + 0.25 * technical
        + 0.15 * event_nlp
        + 0.10 * regime
    )


def _percentile_ranks(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    if len(scores) == 1:
        return {next(iter(scores)): 0.5}

    items = sorted((symbol, _finite(score, 0.0)) for symbol, score in scores.items())
    items.sort(key=lambda item: item[1])
    out: dict[str, float] = {}
    n = len(items)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(items[j + 1][1] - items[i][1]) <= 1e-12:
            j += 1
        avg_position = (i + j) / 2.0
        rank = avg_position / (n - 1)
        for k in range(i, j + 1):
            out[items[k][0]] = rank
        i = j + 1
    return out


def _finite_bucket(values: Mapping[str, float], keys: tuple[str, ...]) -> float:
    if not keys:
        return 1.0
    count = sum(1 for key in keys if _is_finite(values.get(key)))
    return _clamp01(count / len(keys))


def _non_zero_bucket(values: Mapping[str, float], keys: tuple[str, ...]) -> float:
    if not keys:
        return 1.0
    finite_values = [_finite(values.get(key), 0.0) for key in keys if _is_finite(values.get(key))]
    if not finite_values:
        return 0.0
    populated = sum(1 for value in finite_values if abs(value) > 1e-12)
    return _clamp01(populated / len(keys))


def _get(values: Mapping[str, float], key: str, default: float = 0.0) -> float:
    return _finite(values.get(key), default)


def _finite(value: object, default: float) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _is_finite(value: object) -> bool:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(out)


def _linear(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp01((value - low) / (high - low))


def _inverse_z(value: float, bound: float = 3.0) -> float:
    bounded = max(-bound, min(bound, value))
    return _clamp01((bound - bounded) / (2.0 * bound))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sector_key(value: object) -> str:
    text = str(value or "").strip()
    return text or "Unknown"
