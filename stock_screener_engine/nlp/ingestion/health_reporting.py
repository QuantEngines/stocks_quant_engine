"""Ingestion health reporting artifacts for operational monitoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class AdapterIngestionStat:
    adapter_name: str
    source_kind: str
    fetch_count: int
    failure_count: int
    document_count: int
    latency_ms: float


@dataclass
class IngestionHealthSink:
    root_dir: str

    def write_report(
        self,
        *,
        symbols: list[str],
        lookback_days: int,
        adapter_stats: list[AdapterIngestionStat],
    ) -> None:
        now = datetime.now(timezone.utc)
        folder = Path(self.root_dir) / "ingestion_health" / now.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "ingestion_health.jsonl"

        totals = {
            "news": self._aggregate(adapter_stats, "news"),
            "filings": self._aggregate(adapter_stats, "filings"),
        }

        payload = {
            "timestamp_utc": now.isoformat(),
            "symbols": symbols,
            "lookback_days": int(lookback_days),
            "totals": totals,
            "adapters": [
                {
                    "adapter": s.adapter_name,
                    "source_kind": s.source_kind,
                    "fetch_count": s.fetch_count,
                    "failure_count": s.failure_count,
                    "document_count": s.document_count,
                    "latency_ms": round(s.latency_ms, 3),
                }
                for s in adapter_stats
            ],
        }

        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, separators=(",", ":")))
            fh.write("\n")

    def _aggregate(self, stats: list[AdapterIngestionStat], source_kind: str) -> dict[str, float | int]:
        relevant = [s for s in stats if s.source_kind == source_kind]
        return {
            "fetch_count": sum(s.fetch_count for s in relevant),
            "failure_count": sum(s.failure_count for s in relevant),
            "document_count": sum(s.document_count for s in relevant),
            "latency_ms": round(sum(s.latency_ms for s in relevant), 3),
        }


def infer_source_kind(adapter_name: str) -> str:
    lower = adapter_name.lower()
    if "news" in lower:
        return "news"
    if "filing" in lower:
        return "filings"
    return "other"
