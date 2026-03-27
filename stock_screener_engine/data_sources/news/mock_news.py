"""Mock news provider for local research and demo workflows."""

from __future__ import annotations

from typing import Sequence

from stock_screener_engine.data_sources.base.interfaces import NewsProvider


class MockNewsProvider(NewsProvider):
    """Returns synthetic news headlines for demo and unit tests.

    Headlines are deterministically generated from the symbol string so tests
    are repeatable without external API calls.
    """

    _HEADLINE_TEMPLATES = [
        "{symbol} reports strong quarterly earnings; analysts upgrade",
        "{symbol} expands capacity with new plant investment",
        "{symbol} promoters increase stake; governance signal positive",
        "{symbol} secures large order from government client",
        "{symbol} faces short-term headwind from raw material costs",
    ]

    def get_recent_news(
        self, symbols: Sequence[str], lookback_days: int = 7
    ) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for symbol in symbols:
            # deterministic subset based on symbol hash to vary coverage
            n = 1 + (sum(ord(c) for c in symbol) % len(self._HEADLINE_TEMPLATES))
            result[symbol] = [
                tpl.format(symbol=symbol)
                for tpl in self._HEADLINE_TEMPLATES[:n]
            ]
        return result

    def get_news_sentiment(self, symbol: str, lookback_days: int = 7) -> float:
        """Return synthetic sentiment in [-1, 1] derived from symbol name."""
        seed = sum(ord(c) for c in symbol) % 20
        return round((seed - 10) / 10.0, 2)  # deterministic range: -1.0 to 0.9
