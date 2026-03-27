"""Reusable normalization helpers for scoring pipelines."""

from __future__ import annotations


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def linear_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp((value - low) / (high - low))


def inverse_linear_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp((high - value) / (high - low))


def threshold_score(value: float, threshold: float, width: float = 1.0) -> float:
    if width <= 0:
        return 1.0 if value >= threshold else 0.0
    return clamp((value - (threshold - width)) / width)


def symmetric_score(value: float, center: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 1.0 if value == center else 0.0
    distance = abs(value - center)
    return clamp(1.0 - (distance / tolerance))
