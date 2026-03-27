from __future__ import annotations

from stock_screener_engine.backtest.calibration import (
    CalibrationDatasetBuilder,
    ModelCalibrator,
    WeightPriorAutoTuner,
)


def test_calibration_metrics_present() -> None:
    builder = CalibrationDatasetBuilder()
    calibrator = ModelCalibrator()

    scores: dict[tuple[str, str], float] = {}
    returns: dict[tuple[str, str, int], float] = {}
    dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
    symbols = ["A", "B", "C", "D", "E"]
    for di, d in enumerate(dates):
        for si, s in enumerate(symbols):
            score = float(si) + (di * 0.1)
            scores[(d, s)] = score
            returns[(d, s, 1)] = score / 10.0
            returns[(d, s, 5)] = score / 20.0

    rows = builder.build(scores, returns, horizons=[1, 5])
    report = calibrator.evaluate(rows, horizons=[1, 5])

    assert 1 in report.quantile_ic
    assert 5 in report.quantile_ic
    assert 1 in report.turnover_top_quantile
    assert 5 in report.decay


def test_perfect_rank_order_gives_positive_ic() -> None:
    builder = CalibrationDatasetBuilder()
    calibrator = ModelCalibrator()

    scores = {
        ("2025-01-01", "A"): 0.1,
        ("2025-01-01", "B"): 0.2,
        ("2025-01-01", "C"): 0.3,
        ("2025-01-01", "D"): 0.4,
        ("2025-01-01", "E"): 0.5,
    }
    returns = {
        ("2025-01-01", "A", 1): 0.01,
        ("2025-01-01", "B", 1): 0.02,
        ("2025-01-01", "C", 1): 0.03,
        ("2025-01-01", "D", 1): 0.04,
        ("2025-01-01", "E", 1): 0.05,
    }

    rows = builder.build(scores, returns, horizons=[1])
    report = calibrator.evaluate(rows, horizons=[1])
    assert report.quantile_ic[1] > 0.9


def test_weight_prior_auto_tuner_preserves_totals() -> None:
    builder = CalibrationDatasetBuilder()
    calibrator = ModelCalibrator()

    scores = {
        ("2025-01-01", "A"): 0.1,
        ("2025-01-01", "B"): 0.2,
        ("2025-01-01", "C"): 0.3,
        ("2025-01-01", "D"): 0.4,
        ("2025-01-01", "E"): 0.5,
    }
    returns = {
        ("2025-01-01", "A", 1): 0.01,
        ("2025-01-01", "B", 1): 0.02,
        ("2025-01-01", "C", 1): 0.03,
        ("2025-01-01", "D", 1): 0.04,
        ("2025-01-01", "E", 1): 0.05,
    }
    rows = builder.build(scores, returns, horizons=[1])
    report = calibrator.evaluate(rows, horizons=[1])

    base_long = {
        "growth_quality": 18.0,
        "profitability_quality": 17.0,
        "balance_sheet_health": 15.0,
        "cash_flow_quality": 12.0,
        "valuation_sanity": 12.0,
        "governance_proxy": 10.0,
        "event_catalyst": 8.0,
        "regime_tailwind": 8.0,
    }
    base_swing = {
        "trend_strength": 20.0,
        "momentum_strength": 18.0,
        "relative_strength_proxy": 14.0,
        "volatility_regime": 12.0,
        "volume_confirmation": 12.0,
        "event_catalyst": 12.0,
        "sentiment_score": 12.0,
    }

    tuned = WeightPriorAutoTuner().tune(report, base_long, base_swing)
    assert abs(sum(tuned.long_term.values()) - sum(base_long.values())) < 1e-6
    assert abs(sum(tuned.swing.values()) - sum(base_swing.values())) < 1e-6
