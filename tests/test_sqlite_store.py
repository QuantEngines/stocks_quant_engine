from __future__ import annotations

import sqlite3

from stock_screener_engine.core.entities import SignalExplanation, SignalResult
from stock_screener_engine.storage.sqlite_store import SQLiteStore


def test_insert_signals_replaces_existing_same_day_signal_family(tmp_path) -> None:
    store = SQLiteStore(str(tmp_path / "signals.db"))
    try:
        store.insert_signals([
            _signal("AAA", "long_term_reject", 10.0, "long_term"),
            _signal("AAA", "swing_reject", 30.0, "swing"),
        ])
        store.insert_signals([_signal("AAA", "swing_candidate", 55.0, "swing")])

        conn = sqlite3.connect(tmp_path / "signals.db")
        rows = conn.execute(
            "SELECT category FROM signals WHERE symbol = 'AAA' ORDER BY category"
        ).fetchall()
        conn.close()

        assert [row[0] for row in rows] == ["long_term_reject", "swing_candidate"]
    finally:
        store.close()


def _signal(symbol: str, category: str, score: float, signal_type: str) -> SignalResult:
    return SignalResult(
        symbol=symbol,
        category=category,
        score=score,
        explanation=SignalExplanation(signal_type=signal_type, score=score),
    )
