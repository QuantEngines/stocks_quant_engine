"""Simple unstructured-text adapter for sentiment and event placeholders."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from stock_screener_engine.data_sources.base.interfaces import TextEventProvider


class MockTextEventProvider(TextEventProvider):
    def __init__(self) -> None:
        self._events = {
            "RELIANCE": ["Capex expansion announced", "Positive refining margin outlook"],
            "TCS": ["Large deal wins", "Management commentary optimistic"],
            "INFY": ["Muted guidance concern", "Margin recovery expected"],
            "HDFCBANK": ["Stable deposit growth"],
            "LT": ["Order book acceleration"],
            "SBIN": ["Credit cost normalization"],
            "ITC": ["Defensive earnings visibility"],
            "SUNPHARMA": ["USFDA observations resolved"],
        }

    def get_recent_events(self, symbols: Sequence[str], lookback_days: int = 30) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for symbol in symbols:
            result[symbol] = self._events.get(symbol, [])
        return dict(result)

    def get_sentiment_score(self, symbol: str, lookback_days: int = 30) -> float:
        events = self._events.get(symbol, [])
        if not events:
            return 0.0
        positive_tokens = ("positive", "optimistic", "acceleration", "resolved", "wins", "stable")
        negative_tokens = ("concern", "risk", "downgrade", "delay")
        pos = sum(any(tok in e.lower() for tok in positive_tokens) for e in events)
        neg = sum(any(tok in e.lower() for tok in negative_tokens) for e in events)
        score = (pos - neg) / max(1, len(events))
        return max(-1.0, min(1.0, score))
