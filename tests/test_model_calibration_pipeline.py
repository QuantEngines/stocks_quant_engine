from __future__ import annotations

from pathlib import Path

from stock_screener_engine.config.settings import load_settings
from stock_screener_engine.pipelines.model_calibration import ModelCalibrationPipeline


def test_model_calibration_pipeline_writes_artifacts(tmp_path: Path) -> None:
    settings = load_settings()
    settings = settings.__class__(
        environment=settings.environment,
        log_level=settings.log_level,
        storage=settings.storage.__class__(
            root_dir=str(tmp_path),
            sqlite_path=str(tmp_path / "meta.db"),
        ),
        features=settings.features,
        nlp=settings.nlp,
        llm=settings.llm,
        runtime_data=settings.runtime_data,
        integrations=settings.integrations,
        scoring=settings.scoring,
    )

    p = ModelCalibrationPipeline(settings=settings)

    scores = {
        ("2025-01-01", "A"): 0.1,
        ("2025-01-01", "B"): 0.2,
        ("2025-01-01", "C"): 0.3,
        ("2025-01-01", "D"): 0.4,
        ("2025-01-01", "E"): 0.5,
    }
    rets = {
        ("2025-01-01", "A", 1): 0.01,
        ("2025-01-01", "B", 1): 0.02,
        ("2025-01-01", "C", 1): 0.03,
        ("2025-01-01", "D", 1): 0.04,
        ("2025-01-01", "E", 1): 0.05,
    }

    _ = p.run(scores_by_date_symbol=scores, returns_by_date_symbol_horizon=rets, horizons=[1], run_label="t")

    assert (tmp_path / "calibration" / "calibration_dataset_t.csv").exists()
    assert (tmp_path / "calibration" / "calibration_report_t.json").exists()
    assert (tmp_path / "calibration" / "calibration_priors_t.json").exists()
