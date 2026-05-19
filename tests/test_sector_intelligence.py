from __future__ import annotations

from stock_screener_engine.config.settings import load_settings
from stock_screener_engine.core.engine import ResearchEngine
from stock_screener_engine.data_sources.market.mock_market_data import MockIndianMarketDataProvider
from stock_screener_engine.data_sources.text.mock_text_adapter import MockTextEventProvider
from stock_screener_engine.sector.sector_report import SectorIntelligenceBuilder


def test_sector_intelligence_reports_rank_and_stance() -> None:
    settings = load_settings()
    output = ResearchEngine(
        settings=settings,
        market_data=MockIndianMarketDataProvider(),
        text_data=MockTextEventProvider(),
    ).run(regime_score=0.0)

    reports = SectorIntelligenceBuilder().build_from_engine_output(output)

    assert reports
    first = reports[0].to_dict()
    assert "sector_score" in first
    assert first["stance"] in {"overweight", "neutral", "underweight"}
    assert first["coverage"]["stock_count"] > 0


def test_sector_report_renders_markdown() -> None:
    settings = load_settings()
    output = ResearchEngine(
        settings=settings,
        market_data=MockIndianMarketDataProvider(),
        text_data=MockTextEventProvider(),
    ).run(regime_score=0.0)
    reports = SectorIntelligenceBuilder().build_from_engine_output(output)

    markdown = SectorIntelligenceBuilder().render_markdown(reports)

    assert "# Sector Intelligence" in markdown
    assert "Stance:" in markdown
