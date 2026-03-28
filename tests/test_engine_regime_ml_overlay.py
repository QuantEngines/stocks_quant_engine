from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from stock_screener_engine.config.settings import StorageSettings, load_settings
from stock_screener_engine.core.engine import ResearchEngine
from stock_screener_engine.core.ml_ranking import LinearRankModel
from stock_screener_engine.data_sources.market.mock_market_data import MockIndianMarketDataProvider
from stock_screener_engine.data_sources.text.mock_text_adapter import MockTextEventProvider


def test_engine_auto_regime_snapshot_is_present() -> None:
    settings = load_settings()
    engine = ResearchEngine(
        settings=settings,
        market_data=MockIndianMarketDataProvider(),
        text_data=MockTextEventProvider(),
    )

    out = engine.run(regime_score=None)
    assert "regime_snapshot" in out
    assert out["regime_snapshot"]["label"] in {"bull", "bear", "neutral"}


def test_engine_ml_overlay_loads_when_model_file_exists(tmp_path: Path) -> None:
    settings = load_settings()
    settings = replace(
        settings,
        storage=StorageSettings(
            root_dir=str(tmp_path),
            sqlite_path=str(tmp_path / "metadata.db"),
        ),
    )

    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)

    model = LinearRankModel(
        feature_weights={"earnings_growth": 0.7, "relative_strength": 0.3},
        feature_means={"earnings_growth": 0.0, "relative_strength": 0.0},
        feature_stds={"earnings_growth": 1.0, "relative_strength": 1.0},
    )
    (calibration_dir / "ml_rank_model_latest.json").write_text(
        json.dumps(model.to_payload()), encoding="utf-8"
    )

    engine = ResearchEngine(
        settings=settings,
        market_data=MockIndianMarketDataProvider(),
        text_data=MockTextEventProvider(),
    )
    out = engine.run(regime_score=None)

    assert "ml_ranked" in out
    assert len(out["ml_ranked"]) > 0
    assert len(out["ml_long_ranked"]) == len(out["scores"])
