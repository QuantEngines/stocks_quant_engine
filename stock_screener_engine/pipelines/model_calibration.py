"""Model calibration pipeline with persisted dataset and metrics artifacts."""

from __future__ import annotations

from dataclasses import asdict

from stock_screener_engine.backtest.calibration import (
    CalibrationDatasetBuilder,
    CalibrationReport,
    ModelCalibrator,
    WeightPriorAutoTuner,
)
from stock_screener_engine.config.settings import AppSettings
from stock_screener_engine.storage.local_files import LocalFileStorage


class ModelCalibrationPipeline:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.file_store = LocalFileStorage(settings.storage.root_dir)
        self.builder = CalibrationDatasetBuilder()
        self.calibrator = ModelCalibrator()

    def run(
        self,
        scores_by_date_symbol: dict[tuple[str, str], float],
        returns_by_date_symbol_horizon: dict[tuple[str, str, int], float],
        horizons: list[int],
        run_label: str = "latest",
    ) -> CalibrationReport:
        rows = self.builder.build(
            scores_by_date_symbol=scores_by_date_symbol,
            returns_by_date_symbol_horizon=returns_by_date_symbol_horizon,
            horizons=horizons,
        )
        report = self.calibrator.evaluate(rows, horizons=horizons)

        dataset_rows = [
            {
                "as_of": row.as_of,
                "symbol": row.symbol,
                "score": row.score,
                **{f"ret_h{h}": row.returns.get(h, 0.0) for h in horizons},
            }
            for row in rows
        ]
        report_payload = {
            "horizons": horizons,
            "report": asdict(report),
            "row_count": len(rows),
        }

        base_long = {
            k: float(getattr(self.settings.scoring.long_term_weights, k))
            for k in self.settings.scoring.long_term_weights.__dataclass_fields__
        }
        base_swing = {
            k: float(getattr(self.settings.scoring.swing_weights, k))
            for k in self.settings.scoring.swing_weights.__dataclass_fields__
        }
        tuned = WeightPriorAutoTuner().tune(
            report=report,
            long_term_weights=base_long,
            swing_weights=base_swing,
            learning_rate=self.settings.scoring.calibration_auto_tune.learning_rate,
        )
        priors_payload = {
            "horizons": horizons,
            "base_long_term": base_long,
            "base_swing": base_swing,
            "tuned": asdict(tuned),
        }

        self.file_store.save_rows_csv(
            dataset_rows,
            filename=f"calibration_dataset_{run_label}.csv",
            subdir="calibration",
        )
        self.file_store.save_json(
            report_payload,
            filename=f"calibration_report_{run_label}.json",
            subdir="calibration",
        )
        self.file_store.save_json(
            priors_payload,
            filename=f"calibration_priors_{run_label}.json",
            subdir="calibration",
        )

        return report
