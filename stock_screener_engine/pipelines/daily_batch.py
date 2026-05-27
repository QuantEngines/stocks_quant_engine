"""Daily EOD screener pipeline."""

from __future__ import annotations

import logging
from typing import cast

from stock_screener_engine.config.settings import AppSettings
from stock_screener_engine.core.entities import FeatureVector, ScoreCard, SignalResult
from stock_screener_engine.core.engine import ResearchEngine
from stock_screener_engine.data_sources.base.interfaces import (
    FinancialsProvider,
    MarketDataProvider,
    TextEventProvider,
)
from stock_screener_engine.storage.local_files import LocalFileStorage
from stock_screener_engine.storage.sqlite_store import SQLiteStore
from stock_screener_engine.nlp.event_engine.pipeline import TextIntelligencePipeline
from stock_screener_engine.pipelines.quality_reporting import build_pipeline_quality_report

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

    def run(self, symbols: list[str] | None = None) -> dict[str, object]:
        logger.info("Running daily batch pipeline")
        output = self.engine.run(symbols=symbols, regime_score=None)
        quality_report = build_pipeline_quality_report(output, "daily_batch")
        self.file_store.save_json(quality_report, filename="daily_quality_report.json", subdir="quality")
        if not quality_report["passed"]:
            raise RuntimeError("Daily batch blocked by data quality issues")

        features = cast(list[FeatureVector], output["features"])
        scores = cast(list[ScoreCard], output["scores"])
        long_signals = cast(list[SignalResult], output["long_signals"])
        swing_signals = cast(list[SignalResult], output["swing_signals"])

        self.file_store.save_features(features, filename="daily_features.csv")
        self.file_store.save_signals(long_signals, filename="daily_long_signals.json")
        self.file_store.save_signals(swing_signals, filename="daily_swing_signals.json")

        self.sqlite.upsert_features(features)
        self.sqlite.upsert_scores(scores)
        self.sqlite.insert_signals(long_signals)
        self.sqlite.insert_signals(swing_signals)

        return output

    def close(self) -> None:
        self.sqlite.close()
