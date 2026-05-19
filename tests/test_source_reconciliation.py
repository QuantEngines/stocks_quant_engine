from __future__ import annotations

from stock_screener_engine.data_sources.schemas import OHLCVBar
from stock_screener_engine.monitoring.source_reconciliation import SourceReconciler


def test_source_reconciler_passes_close_sources() -> None:
    bars = [
        OHLCVBar("NSE", "AAA", "2026-01-01", 100.0, 101.0, 99.0, 100.0, 1000.0),
        OHLCVBar("BSE", "AAA", "2026-01-01", 100.1, 101.1, 99.1, 100.2, 1100.0),
    ]

    report = SourceReconciler().reconcile(bars, requested_symbols=["AAA"])

    assert report.passed is True
    assert report.metrics["multi_source_groups"] == 1


def test_source_reconciler_flags_price_divergence_and_missing_symbol() -> None:
    bars = [
        OHLCVBar("NSE", "AAA", "2026-01-01", 100.0, 101.0, 99.0, 100.0, 1000.0),
        OHLCVBar("BSE", "AAA", "2026-01-01", 103.0, 104.0, 102.0, 103.0, 1000.0),
    ]

    report = SourceReconciler(max_close_divergence_pct=1.0).reconcile(
        bars,
        requested_symbols=["AAA", "BBB"],
    )

    assert report.passed is False
    assert any("Close divergence" in issue.message for issue in report.issues)
    assert any(issue.symbol == "BBB" for issue in report.issues)
