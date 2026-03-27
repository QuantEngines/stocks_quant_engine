"""Persistent audit logging for low-confidence LLM extraction outcomes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class LowConfidenceAuditSink:
    root_dir: str

    def write(
        self,
        *,
        task: str,
        symbol: str,
        document_id: str,
        confidence: float,
        threshold: float,
        provider: str,
        model: str,
        used_fallback: bool,
        title: str,
        source: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        folder = Path(self.root_dir) / "llm_audit" / now.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "low_confidence.jsonl"

        record = {
            "timestamp_utc": now.isoformat(),
            "task": task,
            "symbol": symbol,
            "document_id": document_id,
            "confidence": round(float(confidence), 6),
            "threshold": round(float(threshold), 6),
            "provider": provider,
            "model": model,
            "used_fallback": used_fallback,
            "title": title[:240],
            "source": source,
        }
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")))
            fh.write("\n")
