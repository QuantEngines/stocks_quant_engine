"""Signal drift monitoring placeholder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DriftReport:
    metric: str
    value: float
    threshold: float
    drifted: bool


class SignalDriftMonitor:
    """Basic placeholder to track score distribution drift over time."""

    def check_mean_shift(self, historical_mean: float, current_mean: float, threshold: float = 8.0) -> DriftReport:
        shift = abs(current_mean - historical_mean)
        return DriftReport(metric="mean_score_shift", value=shift, threshold=threshold, drifted=shift > threshold)
