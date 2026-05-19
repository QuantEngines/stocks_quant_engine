"""Data quality checks for market data, snapshots, and computed features."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Sequence

from stock_screener_engine.core.entities import FeatureVector, StockSnapshot
from stock_screener_engine.data_sources.schemas import OHLCVBar


@dataclass(frozen=True)
class DataQualityReport:
    passed: bool
    issues: list[str]
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float | int | str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }


class DataQualityChecker:
    def validate_snapshots(
        self,
        snapshots: list[StockSnapshot],
        requested_symbols: Sequence[str] | None = None,
        min_coverage: float = 0.8,
        max_staleness_days: int = 7,
        reference_date: date | None = None,
    ) -> DataQualityReport:
        issues: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, float | int | str | None] = {
            "row_count": len(snapshots),
            "unique_symbols": len({s.symbol for s in snapshots}),
        }
        if not snapshots:
            issues.append("No snapshots available")

        requested = _normalise_symbols(requested_symbols or [])
        observed = _normalise_symbols([s.symbol for s in snapshots])
        if requested:
            missing = sorted(requested - observed)
            coverage = len(requested - set(missing)) / len(requested)
            metrics["coverage"] = round(coverage, 4)
            metrics["missing_symbols"] = len(missing)
            if coverage < min_coverage:
                issues.append(
                    f"Snapshot coverage below threshold: {coverage:.1%} < {min_coverage:.1%}"
                )
            if missing:
                warnings.append(f"Missing snapshots for: {', '.join(missing[:10])}")

        duplicates = _duplicates([s.symbol for s in snapshots])
        if duplicates:
            issues.append(f"Duplicate snapshots for: {', '.join(duplicates[:10])}")

        anchor = reference_date or date.today()
        for snap in snapshots:
            if not snap.symbol.strip():
                issues.append("Snapshot with empty symbol")
            if snap.as_of > anchor + timedelta(days=1):
                issues.append(f"{snap.symbol}: snapshot date is in the future")
            if snap.as_of < anchor - timedelta(days=max_staleness_days):
                issues.append(f"{snap.symbol}: stale snapshot as_of={snap.as_of.isoformat()}")
            if snap.close <= 0:
                issues.append(f"{snap.symbol}: non-positive close")
            if snap.volume < 0:
                issues.append(f"{snap.symbol}: negative volume")
            if not (0 <= snap.delivery_ratio <= 1.5):
                issues.append(f"{snap.symbol}: delivery_ratio out of expected range")
            if not snap.sector.strip() or snap.sector.strip().lower() == "unknown":
                warnings.append(f"{snap.symbol}: missing or unknown sector")
        return DataQualityReport(passed=not issues, issues=issues, warnings=warnings, metrics=metrics)

    def validate_features(
        self,
        features: list[FeatureVector],
        expected_symbols: Sequence[str] | None = None,
        min_coverage: float = 0.8,
    ) -> DataQualityReport:
        issues: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, float | int | str | None] = {
            "row_count": len(features),
            "unique_symbols": len({f.symbol for f in features}),
        }
        expected = _normalise_symbols(expected_symbols or [])
        observed = _normalise_symbols([f.symbol for f in features])
        if expected:
            missing = sorted(expected - observed)
            coverage = len(expected - set(missing)) / len(expected)
            metrics["coverage"] = round(coverage, 4)
            metrics["missing_symbols"] = len(missing)
            if coverage < min_coverage:
                issues.append(
                    f"Feature coverage below threshold: {coverage:.1%} < {min_coverage:.1%}"
                )
            if missing:
                warnings.append(f"Missing features for: {', '.join(missing[:10])}")

        duplicates = _duplicates([f.symbol for f in features])
        if duplicates:
            issues.append(f"Duplicate feature vectors for: {', '.join(duplicates[:10])}")

        for fv in features:
            if not fv.values:
                issues.append(f"{fv.symbol}: empty feature vector")
                continue
            for name, value in fv.values.items():
                if not isinstance(value, (int, float)):
                    issues.append(f"{fv.symbol}: non-numeric feature {name}")
                    continue
                if not math.isfinite(float(value)):
                    issues.append(f"{fv.symbol}: non-finite feature {name}")
        return DataQualityReport(passed=not issues, issues=issues, warnings=warnings, metrics=metrics)

    def validate_ohlcv_bars(
        self,
        bars: list[OHLCVBar],
        requested_symbols: Sequence[str] | None = None,
        min_coverage: float = 0.8,
    ) -> DataQualityReport:
        issues: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, float | int | str | None] = {
            "row_count": len(bars),
            "unique_symbols": len({b.symbol for b in bars}),
            "unique_venues": len({b.venue for b in bars}),
        }
        if not bars:
            issues.append("No OHLCV bars available")

        requested = _normalise_symbols(requested_symbols or [])
        observed = _normalise_symbols([b.symbol for b in bars])
        if requested:
            missing = sorted(requested - observed)
            coverage = len(requested - set(missing)) / len(requested)
            metrics["coverage"] = round(coverage, 4)
            metrics["missing_symbols"] = len(missing)
            if coverage < min_coverage:
                issues.append(f"OHLCV coverage below threshold: {coverage:.1%} < {min_coverage:.1%}")
            if missing:
                warnings.append(f"Missing OHLCV bars for: {', '.join(missing[:10])}")

        duplicate_keys = _duplicates([f"{b.venue}:{b.symbol}:{b.ts}" for b in bars])
        if duplicate_keys:
            issues.append(f"Duplicate OHLCV bars for: {', '.join(duplicate_keys[:10])}")

        for bar in bars:
            prefix = f"{bar.venue}:{bar.symbol}:{bar.ts}"
            prices = [bar.open, bar.high, bar.low, bar.close]
            if not bar.symbol.strip():
                issues.append("OHLCV bar with empty symbol")
            if not bar.ts.strip():
                issues.append(f"{prefix}: empty timestamp")
            if not all(math.isfinite(float(p)) for p in prices):
                issues.append(f"{prefix}: non-finite price")
                continue
            if not math.isfinite(float(bar.volume)):
                issues.append(f"{prefix}: non-finite volume")
            if bar.volume < 0:
                issues.append(f"{prefix}: negative volume")
            if min(prices) <= 0:
                issues.append(f"{prefix}: non-positive price")
            if bar.high < bar.low:
                issues.append(f"{prefix}: high below low")
            if bar.open and not (bar.low <= bar.open <= bar.high):
                warnings.append(f"{prefix}: open outside high/low range")
            if bar.close and not (bar.low <= bar.close <= bar.high):
                warnings.append(f"{prefix}: close outside high/low range")

        return DataQualityReport(passed=not issues, issues=issues, warnings=warnings, metrics=metrics)


def _normalise_symbols(symbols: Sequence[str]) -> set[str]:
    return {s.strip().upper() for s in symbols if s and s.strip()}


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        key = value.strip().upper()
        if not key:
            continue
        if key in seen:
            repeated.add(key)
        seen.add(key)
    return sorted(repeated)
