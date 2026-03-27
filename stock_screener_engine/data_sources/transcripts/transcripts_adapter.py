"""Adapter for transcript-like content from any provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from stock_screener_engine.nlp.schemas.events import NormalizedDocument, SourceType


class TranscriptProvider(ABC):
    @abstractmethod
    def get_recent_transcripts(self, symbols: list[str], lookback_days: int = 90) -> dict[str, list[dict]]:
        raise NotImplementedError


class TranscriptsAdapter:
    def __init__(self, provider: TranscriptProvider) -> None:
        self.provider = provider

    def fetch_documents(self, symbols: list[str], lookback_days: int) -> list[NormalizedDocument]:
        rows = self.provider.get_recent_transcripts(symbols, lookback_days=lookback_days)
        out: list[NormalizedDocument] = []
        for symbol in symbols:
            transcripts = rows.get(symbol, [])
            for idx, t in enumerate(transcripts):
                ts = _parse_ts(str(t.get("timestamp", "")))
                title = str(t.get("title", "Earnings call transcript"))
                body = str(t.get("body_text", ""))
                out.append(
                    NormalizedDocument(
                        id=f"transcript:{symbol}:{idx}:{abs(hash(title)) % 100000}",
                        source=SourceType.TRANSCRIPT,
                        timestamp=ts,
                        symbol=symbol,
                        company_name=symbol,
                        title=title,
                        body_text=body,
                        source_name="transcript",
                        url="",
                        metadata={"speaker": str(t.get("speaker", "management"))},
                    )
                )
        return out


def _parse_ts(text: str) -> datetime:
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return datetime.utcnow()
