from __future__ import annotations

from datetime import date

from stock_screener_engine.core.entities import FeatureVector, StockSnapshot
from stock_screener_engine.data_sources.schemas import OHLCVBar
from stock_screener_engine.monitoring.data_quality import DataQualityChecker


def test_snapshot_quality_tracks_coverage_duplicates_and_staleness() -> None:
    checker = DataQualityChecker()
    snapshots = [
        _snapshot("AAA", as_of=date(2026, 1, 1)),
        _snapshot("AAA", as_of=date(2026, 1, 1)),
    ]

    report = checker.validate_snapshots(
        snapshots,
        requested_symbols=["AAA", "BBB"],
        min_coverage=1.0,
        max_staleness_days=3,
        reference_date=date(2026, 1, 10),
    )

    assert not report.passed
    assert report.metrics["coverage"] == 0.5
    assert any("coverage" in issue.lower() for issue in report.issues)
    assert any("duplicate" in issue.lower() for issue in report.issues)
    assert any("stale" in issue.lower() for issue in report.issues)
    assert any("BBB" in warning for warning in report.warnings)


def test_feature_quality_rejects_non_finite_values() -> None:
    checker = DataQualityChecker()
    features = [FeatureVector(symbol="AAA", as_of=date(2026, 1, 1), values={"alpha": float("nan")})]

    report = checker.validate_features(features, expected_symbols=["AAA"])

    assert not report.passed
    assert any("non-finite" in issue for issue in report.issues)


def test_ohlcv_quality_rejects_bad_bars() -> None:
    checker = DataQualityChecker()
    bars = [
        OHLCVBar(
            venue="NSE",
            symbol="AAA",
            ts="2026-01-01",
            open=100.0,
            high=90.0,
            low=95.0,
            close=0.0,
            volume=-1.0,
        )
    ]

    report = checker.validate_ohlcv_bars(bars, requested_symbols=["AAA"])

    assert not report.passed
    assert any("negative volume" in issue for issue in report.issues)
    assert any("non-positive price" in issue for issue in report.issues)
    assert any("high below low" in issue for issue in report.issues)


def _snapshot(symbol: str, as_of: date) -> StockSnapshot:
    return StockSnapshot(
        symbol=symbol,
        as_of=as_of,
        sector="IT",
        close=100.0,
        volume=1_000_000.0,
        delivery_ratio=0.5,
        pe_ratio=0.0,
        roe=0.0,
        debt_to_equity=0.0,
        earnings_growth=0.0,
        free_cash_flow_margin=0.0,
        promoter_holding_change=0.0,
        insider_activity_score=0.0,
    )
