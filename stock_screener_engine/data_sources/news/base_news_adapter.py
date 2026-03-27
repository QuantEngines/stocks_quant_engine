"""Provider-agnostic base adapter for news document ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod

from stock_screener_engine.nlp.schemas.events import NormalizedDocument


class BaseNewsAdapter(ABC):
    @abstractmethod
    def fetch_documents(self, symbols: list[str], lookback_days: int) -> list[NormalizedDocument]:
        raise NotImplementedError
