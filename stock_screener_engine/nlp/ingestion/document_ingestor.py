"""Ingestion layer for normalized text documents from multiple adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from stock_screener_engine.nlp.ingestion.health_reporting import (
    AdapterIngestionStat,
    IngestionHealthSink,
    infer_source_kind,
)
from stock_screener_engine.nlp.schemas.events import NormalizedDocument


@dataclass
class TextDocumentIngestor:
    adapters: list[object] = field(default_factory=list)
    health_sink: IngestionHealthSink | None = None

    def ingest(self, symbols: list[str], lookback_days: int = 30) -> list[NormalizedDocument]:
        documents: list[NormalizedDocument] = []
        adapter_stats: list[AdapterIngestionStat] = []
        try:
            for adapter in self.adapters:
                fetch = getattr(adapter, "fetch_documents", None)
                if callable(fetch):
                    name = adapter.__class__.__name__
                    started = perf_counter()
                    rows: list[NormalizedDocument] = []
                    failed = False
                    try:
                        rows = fetch(symbols, lookback_days)
                        documents.extend(rows)
                    except Exception:
                        failed = True
                        raise
                    finally:
                        elapsed_ms = (perf_counter() - started) * 1000.0
                        adapter_stats.append(
                            AdapterIngestionStat(
                                adapter_name=name,
                                source_kind=infer_source_kind(name),
                                fetch_count=1,
                                failure_count=1 if failed else 0,
                                document_count=0 if failed else len(rows),
                                latency_ms=elapsed_ms,
                            )
                        )
        finally:
            if self.health_sink is not None:
                self.health_sink.write_report(
                    symbols=symbols,
                    lookback_days=lookback_days,
                    adapter_stats=adapter_stats,
                )
        return documents
