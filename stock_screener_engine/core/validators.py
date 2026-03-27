"""Validation utilities for feature coverage and scoring readiness."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureCoverage:
    present: int
    missing: int
    invalid: int

    @property
    def completeness(self) -> float:
        total = self.present + self.missing + self.invalid
        if total <= 0:
            return 0.0
        return self.present / total


def coverage_from_lists(required: list[str], missing: list[str], invalid: list[str]) -> FeatureCoverage:
    required_set = set(required)
    miss = required_set.intersection(missing)
    inv = required_set.intersection(invalid)
    present = len(required_set - miss - inv)
    return FeatureCoverage(present=present, missing=len(miss), invalid=len(inv))


def confidence_from_coverage(coverage: FeatureCoverage, floor: float = 0.2, cap: float = 1.0) -> float:
    c = coverage.completeness
    return max(floor, min(cap, c))
