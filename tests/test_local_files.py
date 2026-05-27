from __future__ import annotations

import json

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
