from __future__ import annotations

import json
from datetime import date

from stock_screener_engine.core.entities import SignalExplanation, SignalResult
from stock_screener_engine.storage.local_files import LocalFileStorage


def test_save_signals_includes_sector_in_json_payload(tmp_path) -> None:
    storage = LocalFileStorage(str(tmp_path))
    path = storage.save_signals(
        [
            SignalResult(
                symbol="ITC",
                sector="Fast Moving Consumer Goods",
                category="swing_candidate",
                score=43.4,
                explanation=SignalExplanation(signal_type="swing", score=43.4),
            )
        ]
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload[0]["sector"] == "Fast Moving Consumer Goods"


def test_save_json_serializes_dates(tmp_path) -> None:
    storage = LocalFileStorage(str(tmp_path))
    path = storage.save_json(
        {"quality_issues": [{"as_of": date(2026, 5, 28), "message": "warning"}]},
        filename="report.json",
        subdir="quality",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["quality_issues"][0]["as_of"] == "2026-05-28"


def test_save_text_writes_to_requested_subdir(tmp_path) -> None:
    storage = LocalFileStorage(str(tmp_path))
    path = storage.save_text("# Coverage\n", filename="coverage.md", subdir="quality")

    assert path == tmp_path / "quality" / "coverage.md"
    assert path.read_text(encoding="utf-8") == "# Coverage\n"
