"""Feature refresh pipeline to recompute and persist feature store."""

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

logger = logging.getLogger(__name__)


class FeatureRefreshPipeline:
    def __init__(
        self,
        settings: AppSettings,
        market_data: MarketDataProvider,
        text_data: TextEventProvider,
        financials: FinancialsProvider | None = None,
    ) -> None:
        self.engine = ResearchEngine(
            settings=settings,
            market_data=market_data,
            text_data=text_data,
            financials=financials,
        )
        self.file_store = LocalFileStorage(settings.storage.root_dir)
        self.sqlite = SQLiteStore(settings.storage.sqlite_path)

    def run(self, symbols: list[str] | None = None) -> list:
        logger.info("Refreshing feature store")
        output = self.engine.run(symbols=symbols, regime_score=0.2)
        features = output["features"]
        self.file_store.save_features(features, filename="refreshed_features.csv")
        self.sqlite.upsert_features(features)
        return features

    def close(self) -> None:
        self.sqlite.close()
