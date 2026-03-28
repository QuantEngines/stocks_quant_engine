from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from stock_screener_engine.config.settings import StorageSettings, load_settings
from stock_screener_engine.monitoring.live_invalidation import ActiveSignal
from stock_screener_engine.pipelines.live_invalidation import LiveInvalidationPipeline


def test_live_invalidation_pipeline_persists_reports(tmp_path: Path) -> None:
    settings = load_settings()
    settings = replace(
        settings,
        storage=StorageSettings(
            root_dir=str(tmp_path),
            sqlite_path=str(tmp_path / "metadata.db"),
        ),
    )

    pipeline = LiveInvalidationPipeline(settings)
    signals = [
        ActiveSignal(
            symbol="AAA",
            side="long",
            entry_price=100.0,
            entered_on=date.today() - timedelta(days=5),
            required_thesis_flags=["earnings_momentum"],
        ),
        ActiveSignal(
            symbol="BBB",
            side="short",
            entry_price=200.0,
            entered_on=date.today() - timedelta(days=2),
        ),
    ]

    payload = pipeline.run(
        active_signals=signals,
        latest_price_by_symbol={"AAA": 89.0, "BBB": 198.0},
        thesis_flags_by_symbol={"AAA": []},
        run_label="unit",
    )

    assert payload["total"] == 2
    assert payload["invalidated"] >= 1

    report_json = tmp_path / "signals" / "live_invalidation_unit.json"
    report_csv = tmp_path / "signals" / "live_invalidation_unit.csv"
    assert report_json.exists()
    assert report_csv.exists()
