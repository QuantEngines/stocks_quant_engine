"""No-op transcript provider for deployment environments without transcript feeds."""

from __future__ import annotations

from stock_screener_engine.data_sources.transcripts.transcripts_adapter import TranscriptProvider


class NullTranscriptProvider(TranscriptProvider):
    def get_recent_transcripts(self, symbols: list[str], lookback_days: int = 90) -> dict[str, list[dict]]:
        return {symbol: [] for symbol in symbols}
