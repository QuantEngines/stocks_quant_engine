"""Lightweight ML-style ranking without external dependencies.

This module provides a deterministic linear ranker trained from calibration
samples. It is intentionally simple so it can run in restricted research
environments where heavy ML libraries may not be available.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping


@dataclass(frozen=True)
class CalibrationSample:
    """Single row for fitting feature-return relationships."""

    symbol: str
    features: Mapping[str, float]
    forward_return: float


@dataclass(frozen=True)
class LinearRankModel:
    """Learned linear model that maps feature vectors to alpha scores."""

    feature_weights: Mapping[str, float]
    feature_means: Mapping[str, float]
    feature_stds: Mapping[str, float]

    def score(self, features: Mapping[str, float]) -> float:
        """Return a scalar alpha score for one feature vector."""
        score = 0.0
        for name, weight in self.feature_weights.items():
            raw = float(features.get(name, 0.0))
            mean = float(self.feature_means.get(name, 0.0))
            std = float(self.feature_stds.get(name, 1.0))
            z = (raw - mean) / std if std > 1e-9 else 0.0
            score += weight * z
        return score

    def rank(self, candidates: Mapping[str, Mapping[str, float]]) -> list[tuple[str, float]]:
        """Rank symbols by descending model score."""
        scored = [(symbol, self.score(features)) for symbol, features in candidates.items()]
        return sorted(scored, key=lambda row: (row[1], row[0]), reverse=True)


class LinearMlRanker:
    """Train a linear alpha model via feature-return covariance signals.

    Training approach
    -----------------
    1. Compute z-score normalization stats for each feature.
    2. Estimate each feature's predictive contribution using covariance with
       forward returns.
    3. Convert contributions to a stable weight vector where absolute weights
       sum to 1.
    """

    def fit(self, samples: list[CalibrationSample], feature_names: list[str]) -> LinearRankModel:
        if not samples:
            raise ValueError("samples cannot be empty")
        if not feature_names:
            raise ValueError("feature_names cannot be empty")

        means = {name: self._mean([float(s.features.get(name, 0.0)) for s in samples]) for name in feature_names}
        stds = {
            name: max(self._std([float(s.features.get(name, 0.0)) for s in samples], means[name]), 1e-9)
            for name in feature_names
        }

        returns = [float(s.forward_return) for s in samples]
        returns_mean = self._mean(returns)

        raw_weights: dict[str, float] = {}
        for name in feature_names:
            xs = [float(s.features.get(name, 0.0)) for s in samples]
            cov = self._covariance(xs, means[name], returns, returns_mean)
            raw_weights[name] = cov / (stds[name] ** 2)

        norm = sum(abs(w) for w in raw_weights.values())
        if norm <= 1e-12:
            uniform = 1.0 / len(feature_names)
            normalized = {name: uniform for name in feature_names}
        else:
            normalized = {name: raw_weights[name] / norm for name in feature_names}

        return LinearRankModel(
            feature_weights=normalized,
            feature_means=means,
            feature_stds=stds,
        )

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values)

    @staticmethod
    def _std(values: list[float], mean: float) -> float:
        if len(values) < 2:
            return 1.0
        var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return sqrt(max(var, 0.0))

    @staticmethod
    def _covariance(xs: list[float], x_mean: float, ys: list[float], y_mean: float) -> float:
        if len(xs) != len(ys) or len(xs) < 2:
            return 0.0
        total = 0.0
        for x, y in zip(xs, ys):
            total += (x - x_mean) * (y - y_mean)
        return total / (len(xs) - 1)
