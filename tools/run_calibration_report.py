"""Generate calibration dataset/report artifacts from a synthetic demo panel.

This script is intended for CI smoke checks and local workflow validation.
"""

from __future__ import annotations

from stock_screener_engine.config.settings import load_settings
from stock_screener_engine.pipelines.model_calibration import ModelCalibrationPipeline


def main() -> None:
    settings = load_settings()
    pipeline = ModelCalibrationPipeline(settings=settings)

    dates = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "LT"]
    horizons = [1, 5, 20]

    scores: dict[tuple[str, str], float] = {}
    returns: dict[tuple[str, str, int], float] = {}
    for di, d in enumerate(dates):
        for si, s in enumerate(symbols):
            score = 0.1 * si + 0.01 * di
            scores[(d, s)] = score
            for h in horizons:
                returns[(d, s, h)] = score / max(1, h)

    report = pipeline.run(
        scores_by_date_symbol=scores,
        returns_by_date_symbol_horizon=returns,
        horizons=horizons,
        run_label="demo",
    )
    print("quantile_ic", report.quantile_ic)
    print("turnover_top_quantile", report.turnover_top_quantile)
    print("decay", report.decay)


if __name__ == "__main__":
    main()
