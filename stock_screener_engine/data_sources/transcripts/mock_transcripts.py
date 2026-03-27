"""Mock transcript provider for deterministic NLP pipeline testing."""

from __future__ import annotations

from datetime import datetime

from stock_screener_engine.data_sources.transcripts.transcripts_adapter import TranscriptProvider


class MockTranscriptProvider(TranscriptProvider):
    def get_recent_transcripts(self, symbols: list[str], lookback_days: int = 90) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        now = datetime.utcnow().replace(microsecond=0).isoformat()
        for symbol in symbols:
            out[symbol] = [
                {
                    "timestamp": now,
                    "title": f"{symbol} management commentary",
                    "body_text": f"Management remains optimistic on demand and guidance for {symbol}",
                    "speaker": "management",
                }
            ]
        return out
