"""Signal drift monitoring — distribution-level and IC-trend diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DriftReport:
    metric:    str
    value:     float
    threshold: float
    drifted:   bool


@dataclass(frozen=True)
class ScoreDistributionSnapshot:
    """Compact representation of a score distribution for drift comparison."""

    date:    str
    mean:    float
    std:     float
    p25:     float
    median:  float
    p75:     float
    n:       int

    @classmethod
    def from_scores(cls, date: str, scores: list[float]) -> "ScoreDistributionSnapshot":
        n = len(scores)
        if n == 0:
            return cls(date=date, mean=0.0, std=0.0, p25=0.0, median=0.0, p75=0.0, n=0)
        s = sorted(scores)
        mean = sum(s) / n
        std  = math.sqrt(sum((x - mean) ** 2 for x in s) / n) if n > 1 else 0.0
        return cls(
            date=date,
            mean=mean,
            std=std,
            p25=_percentile(s, 0.25),
            median=_percentile(s, 0.50),
            p75=_percentile(s, 0.75),
            n=n,
        )


@dataclass
class SignalDriftMonitor:
    """Track score distribution shifts over rolling windows.

    Usage
    -----
    monitor = SignalDriftMonitor()
    monitor.record("2025-01-01", scores_jan)
    monitor.record("2025-02-01", scores_feb)
    reports = monitor.check_drift()
    """

    snapshots: list[ScoreDistributionSnapshot] = field(default_factory=list)
    mean_shift_threshold: float  = 8.0
    std_shift_threshold:  float  = 5.0
    coverage_threshold:   float  = 0.10  # 10 pp change in useful-signal fraction

    def record(self, date: str, scores: list[float]) -> ScoreDistributionSnapshot:
        snap = ScoreDistributionSnapshot.from_scores(date, scores)
        self.snapshots.append(snap)
        return snap

    def check_mean_shift(
        self,
        historical_mean: float,
        current_mean: float,
        threshold: float | None = None,
    ) -> DriftReport:
        t = threshold if threshold is not None else self.mean_shift_threshold
        shift = abs(current_mean - historical_mean)
        return DriftReport(metric="mean_score_shift", value=shift, threshold=t, drifted=shift > t)

    def check_drift(self, lookback: int = 4) -> list[DriftReport]:
        """Compare latest snapshot against the lookback-window average."""
        if len(self.snapshots) < 2:
            return []

        recent   = self.snapshots[-1]
        baseline = self.snapshots[max(0, len(self.snapshots) - 1 - lookback) : -1]
        if not baseline:
            return []

        b_mean   = sum(s.mean   for s in baseline) / len(baseline)
        b_std    = sum(s.std    for s in baseline) / len(baseline)
        b_median = sum(s.median for s in baseline) / len(baseline)

        return [
            DriftReport(
                metric="mean_shift",
                value=abs(recent.mean - b_mean),
                threshold=self.mean_shift_threshold,
                drifted=abs(recent.mean - b_mean) > self.mean_shift_threshold,
            ),
            DriftReport(
                metric="std_shift",
                value=abs(recent.std - b_std),
                threshold=self.std_shift_threshold,
                drifted=abs(recent.std - b_std) > self.std_shift_threshold,
            ),
            DriftReport(
                metric="median_shift",
                value=abs(recent.median - b_median),
                threshold=self.mean_shift_threshold,
                drifted=abs(recent.median - b_median) > self.mean_shift_threshold,
            ),
        ]


def _percentile(sorted_values: list[float], q: float) -> float:
    n = len(sorted_values)
    if n == 0:
        return 0.0
    idx = (n - 1) * q
    lo  = int(idx)
    hi  = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac
