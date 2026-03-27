"""Data quality checks for snapshots and computed features."""

from __future__ import annotations

from dataclasses import dataclass

from stock_screener_engine.core.entities import FeatureVector, StockSnapshot


@dataclass(frozen=True)
class DataQualityReport:
    passed: bool
    issues: list[str]


class DataQualityChecker:
    def validate_snapshots(self, snapshots: list[StockSnapshot]) -> DataQualityReport:
        issues: list[str] = []
        if not snapshots:
            issues.append("No snapshots available")
        for snap in snapshots:
            if snap.close <= 0:
                issues.append(f"{snap.symbol}: non-positive close")
            if snap.volume < 0:
                issues.append(f"{snap.symbol}: negative volume")
            if not (0 <= snap.delivery_ratio <= 1.5):
                issues.append(f"{snap.symbol}: delivery_ratio out of expected range")
        return DataQualityReport(passed=not issues, issues=issues)

    def validate_features(self, features: list[FeatureVector]) -> DataQualityReport:
        issues: list[str] = []
        for fv in features:
            if not fv.values:
                issues.append(f"{fv.symbol}: empty feature vector")
        return DataQualityReport(passed=not issues, issues=issues)
