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

def percentile_rank(value: float, population: list[float]) -> float:
    """Return the percentile rank of ``value`` within ``population`` [0, 1].

    Useful for computing cross-sectional relative ranks (e.g. sector PE rank).
    Returns 0.5 when population is empty (neutral).
    """
    n = len(population)
    if n == 0:
        return 0.5
    below = sum(1 for x in population if x < value)
    return below / n


def log_score(value: float, low: float, high: float) -> float:
    """Like linear_score but uses a log scale for right-skewed distributions.

    Useful for volume, market-cap, or any feature with a long right tail.
    Returns 0 when value <= low; 1 when value >= high.
    """
    import math
    if high <= low or low <= 0:
        return 0.0
    log_lo  = math.log(max(1e-12, low))
    log_hi  = math.log(max(1e-12, high))
    log_val = math.log(max(1e-12, value))
    return clamp((log_val - log_lo) / (log_hi - log_lo))
