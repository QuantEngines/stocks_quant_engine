"""Daily EOD screener pipeline."""

from __future__ import annotations

import logging

from stock_screener_engine.config.settings import AppSettings
from stock_screener_engine.core.engine import ResearchEngine
from stock_screener_engine.data_sources.base.interfaces import (
    FinancialsProvider,
    MarketDataProvider,
    TextEventProvider,
)
from stock_screener_engine.storage.local_files import LocalFileStorage
from stock_screener_engine.storage.sqlite_store import SQLiteStore
from stock_screener_engine.nlp.event_engine.pipeline import TextIntelligencePipeline

logger = logging.getLogger(__name__)


class DailyBatchPipeline:
    def __init__(
        self,
        settings: AppSettings,
        market_data: MarketDataProvider,
        text_data: TextEventProvider,
        financials: FinancialsProvider | None = None,
        text_pipeline: TextIntelligencePipeline | None = None,
    ) -> None:
        self.settings = settings
        self.engine = ResearchEngine(
            settings=settings,
            market_data=market_data,
            text_data=text_data,
            financials=financials,
            text_pipeline=text_pipeline,
        )
        self.file_store = LocalFileStorage(settings.storage.root_dir)
        self.sqlite = SQLiteStore(settings.storage.sqlite_path)

    def run(self, symbols: list[str] | None = None) -> dict[str, list]:
        logger.info("Running daily batch pipeline")
        output = self.engine.run(symbols=symbols, regime_score=0.25)

        self.file_store.save_features(output["features"], filename="daily_features.csv")
        self.file_store.save_signals(output["long_signals"], filename="daily_long_signals.json")
        self.file_store.save_signals(output["swing_signals"], filename="daily_swing_signals.json")

        self.sqlite.upsert_features(output["features"])
        self.sqlite.upsert_scores(output["scores"])
        self.sqlite.insert_signals(output["long_signals"])
        self.sqlite.insert_signals(output["swing_signals"])

        return output

    def close(self) -> None:
        self.sqlite.close()
