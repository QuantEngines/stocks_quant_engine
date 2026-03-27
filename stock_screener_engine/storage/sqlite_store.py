"""SQLite metadata store with upgrade path to Postgres."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from stock_screener_engine.core.entities import FeatureVector, ScoreCard, SignalResult


class SQLiteStore:
    def __init__(self, sqlite_path: str) -> None:
        self.path = Path(sqlite_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                symbol TEXT NOT NULL,
                as_of TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(symbol, as_of)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scores (
                symbol TEXT NOT NULL,
                as_of TEXT NOT NULL,
                long_term_score REAL NOT NULL,
                swing_score REAL NOT NULL,
                risk_penalty REAL NOT NULL,
                conviction REAL NOT NULL,
                components TEXT NOT NULL,
                PRIMARY KEY(symbol, as_of)
            )
            """
        )
        self._ensure_signals_schema(cur)
        self.conn.commit()

    def _ensure_signals_schema(self, cur: sqlite3.Cursor) -> None:
        """Create or migrate the signals table.

        The signals table is an ephemeral output — on schema change (e.g. adding
        ``run_date`` as a dedup key) the table is dropped and recreated.  This is
        safe because signals can always be regenerated from the pipeline.
        """
        cur.execute("PRAGMA table_info(signals)")
        existing_cols = {row[1] for row in cur.fetchall()}
        if existing_cols and "run_date" not in existing_cols:
            cur.execute("DROP TABLE IF EXISTS signals")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                symbol    TEXT NOT NULL,
                category  TEXT NOT NULL,
                run_date  TEXT NOT NULL,
                score     REAL NOT NULL,
                explanation TEXT NOT NULL,
                PRIMARY KEY (symbol, category, run_date)
            )
            """
        )

    def upsert_features(self, vectors: Iterable[FeatureVector]) -> None:
        cur = self.conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO features(symbol, as_of, payload) VALUES (?, ?, ?)",
            [(v.symbol, v.as_of.isoformat(), json.dumps(dict(v.values))) for v in vectors],
        )
        self.conn.commit()

    def upsert_scores(self, cards: Iterable[ScoreCard]) -> None:
        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT OR REPLACE INTO scores(symbol, as_of, long_term_score, swing_score, risk_penalty, conviction, components)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    c.symbol,
                    c.as_of.isoformat(),
                    c.long_term_score,
                    c.swing_score,
                    c.risk_penalty,
                    c.conviction,
                    json.dumps(dict(c.component_scores)),
                )
                for c in cards
            ],
        )
        self.conn.commit()

    def insert_signals(self, signals: Iterable[SignalResult]) -> None:
        from datetime import date as _date

        run_date = _date.today().isoformat()
        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT OR REPLACE INTO signals(symbol, category, run_date, score, explanation)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    s.symbol,
                    s.category,
                    run_date,
                    s.score,
                    __import__("json").dumps(
                        {
                            "signal_type": s.explanation.signal_type,
                            "score": s.explanation.score,
                            "top_positive_drivers": s.explanation.top_positive_drivers,
                            "top_negative_drivers": s.explanation.top_negative_drivers,
                            "ranking_reason": s.explanation.ranking_reason,
                            "rejection_reason": s.explanation.rejection_reason,
                            "holding_horizon": s.explanation.holding_horizon,
                            "risk_flags": s.explanation.risk_flags,
                            "confidence": s.explanation.confidence,
                        }
                    ),
                )
                for s in signals
            ],
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
