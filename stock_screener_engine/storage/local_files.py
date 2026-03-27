"""Raw and cleaned file repositories (CSV/JSON-ready foundation)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from stock_screener_engine.core.entities import FeatureVector, SignalResult


class LocalFileStorage:
    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir)
        self.raw_dir = self.root / "raw"
        self.cleaned_dir = self.root / "cleaned"
        self.feature_dir = self.root / "features"
        self.signal_dir = self.root / "signals"
        self.calibration_dir = self.root / "calibration"
        for path in [
            self.raw_dir,
            self.cleaned_dir,
            self.feature_dir,
            self.signal_dir,
            self.calibration_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def save_raw_payload(self, name: str, payload: dict) -> Path:
        output = self.raw_dir / f"{name}.json"
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output

    def save_features(self, rows: Iterable[FeatureVector], filename: str = "features.csv") -> Path:
        output = self.feature_dir / filename
        rows = list(rows)
        if not rows:
            output.write_text("", encoding="utf-8")
            return output

        all_keys = sorted({k for fv in rows for k in fv.values.keys()})
        with output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["symbol", "as_of", *all_keys])
            writer.writeheader()
            for fv in rows:
                row = {"symbol": fv.symbol, "as_of": fv.as_of.isoformat(), **dict(fv.values)}
                writer.writerow(row)
        return output

    def save_signals(self, rows: Iterable[SignalResult], filename: str = "signals.json") -> Path:
        output = self.signal_dir / filename
        payload = [
            {
                "symbol": s.symbol,
                "category": s.category,
                "score": s.score,
                "explanation": {
                    "signal_type": s.explanation.signal_type,
                    "score": s.explanation.score,
                    "top_positive_drivers": s.explanation.top_positive_drivers,
                    "top_negative_drivers": s.explanation.top_negative_drivers,
                    "ranking_reason": s.explanation.ranking_reason,
                    "rejection_reason": s.explanation.rejection_reason,
                    "holding_horizon": s.explanation.holding_horizon,
                    "risk_flags": s.explanation.risk_flags,
                    "confidence": s.explanation.confidence,
                    "entry_logic": s.explanation.entry_logic,
                    "invalidation_logic": s.explanation.invalidation_logic,
                },
            }
            for s in rows
        ]
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output

    def save_json(self, payload: dict | list, filename: str, subdir: str = "raw") -> Path:
        target_dir = self._resolve_subdir(subdir)
        output = target_dir / filename
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output

    def save_rows_csv(self, rows: list[dict], filename: str, subdir: str = "cleaned") -> Path:
        target_dir = self._resolve_subdir(subdir)
        output = target_dir / filename
        if not rows:
            output.write_text("", encoding="utf-8")
            return output

        columns = sorted({k for row in rows for k in row.keys()})
        with output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return output

    def _resolve_subdir(self, subdir: str) -> Path:
        mapping = {
            "raw": self.raw_dir,
            "cleaned": self.cleaned_dir,
            "features": self.feature_dir,
            "signals": self.signal_dir,
            "calibration": self.calibration_dir,
        }
        if subdir in mapping:
            return mapping[subdir]
        target = self.root / subdir
        target.mkdir(parents=True, exist_ok=True)
        return target
