"""Adapter to bridge existing TextEventProvider into normalized document ingestion."""

from __future__ import annotations

from datetime import datetime, timedelta

from stock_screener_engine.data_sources.base.interfaces import TextEventProvider
from stock_screener_engine.nlp.schemas.events import NormalizedDocument, SourceType


class TextEventAdapter:
    def __init__(self, provider: TextEventProvider) -> None:
        self.provider = provider

    def fetch_documents(self, symbols: list[str], lookback_days: int) -> list[NormalizedDocument]:
        rows = self.provider.get_recent_events(symbols, lookback_days=lookback_days)
        now = datetime.utcnow()
        out: list[NormalizedDocument] = []
        for symbol in symbols:
            events = rows.get(symbol, [])
            for idx, event in enumerate(events):
                out.append(
                    NormalizedDocument(
                        id=f"textevent:{symbol}:{idx}:{abs(hash(event)) % 100000}",
                        source=SourceType.ANNOUNCEMENT,
                        timestamp=now - timedelta(hours=idx * 6),
                        symbol=symbol,
                        company_name=symbol,
                        title=event,
                        body_text=event,
                        source_name=self.provider.__class__.__name__,
                        url="",
                        metadata={"provider": self.provider.__class__.__name__},
                    )
                )
        return out
