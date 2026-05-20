from __future__ import annotations

from datetime import date

from stock_screener_engine.config.settings import load_settings
from stock_screener_engine.core.engine import ResearchEngine
from stock_screener_engine.core.entities import FeatureVector, ScoreCard, SignalExplanation, SignalResult
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


def test_sector_report_deduplicates_avoid_watchlist_symbols() -> None:
    as_of = date(2026, 5, 1)
    explanation = SignalExplanation(signal_type="test", score=10.0)
    reports = SectorIntelligenceBuilder().build(
        features=[
            FeatureVector(
                symbol="AAA",
                as_of=as_of,
                values={"trend_strength": 0.1, "momentum_strength": 0.1},
            )
        ],
        scores=[
            ScoreCard(
                symbol="AAA",
                as_of=as_of,
                long_term_score=1.0,
                swing_score=1.0,
                risk_penalty=10.0,
                conviction=1.0,
                component_scores={},
            )
        ],
        long_signals=[
            SignalResult(
                symbol="AAA",
                category="long_term_reject",
                score=1.0,
                explanation=explanation,
                sector="IT",
            )
        ],
        swing_signals=[
            SignalResult(
                symbol="AAA",
                category="swing_reject",
                score=2.0,
                explanation=explanation,
                sector="IT",
            )
        ],
    )

    assert reports[0].avoid_watchlist_stocks == ["AAA"]
