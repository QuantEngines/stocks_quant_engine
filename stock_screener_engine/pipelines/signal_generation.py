"""Signal generation pipeline for explicit score-to-signal runs."""

from __future__ import annotations

import logging

from stock_screener_engine.config.settings import AppSettings
from stock_screener_engine.core.engine import ResearchEngine
from stock_screener_engine.data_sources.base.interfaces import (
    FinancialsProvider,
    MarketDataProvider,
    TextEventProvider,
)
from stock_screener_engine.nlp.event_engine.pipeline import TextIntelligencePipeline

logger = logging.getLogger(__name__)


class SignalGenerationPipeline:
    def __init__(
        self,
        settings: AppSettings,
        market_data: MarketDataProvider,
        text_data: TextEventProvider,
        financials: FinancialsProvider | None = None,
        text_pipeline: TextIntelligencePipeline | None = None,
    ) -> None:
        self.engine = ResearchEngine(
            settings=settings,
            market_data=market_data,
            text_data=text_data,
            financials=financials,
            text_pipeline=text_pipeline,
        )

    def run(self, symbols: list[str] | None = None) -> dict[str, list]:
        logger.info("Running signal generation pipeline")
        return self.engine.run(symbols=symbols, regime_score=0.2)
