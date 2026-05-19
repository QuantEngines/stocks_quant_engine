"""Intraday update pipeline focused on swing-sensitive updates."""

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
from stock_screener_engine.nlp.event_engine.pipeline import TextIntelligencePipeline
from stock_screener_engine.pipelines.quality_reporting import build_pipeline_quality_report

logger = logging.getLogger(__name__)


class IntradayUpdatePipeline:
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
        self.file_store = LocalFileStorage(settings.storage.root_dir)

    def run(self, symbols: list[str] | None = None) -> dict[str, list]:
        logger.info("Running intraday swing update pipeline")
        output = self.engine.run(symbols=symbols, regime_score=None)
        quality_report = build_pipeline_quality_report(output, "intraday_update")
        self.file_store.save_json(quality_report, filename="intraday_quality_report.json", subdir="quality")
        if not quality_report["passed"]:
            raise RuntimeError("Intraday update blocked by data quality issues")
        self.file_store.save_signals(output["swing_signals"], filename="intraday_swing_signals.json")
        return output
