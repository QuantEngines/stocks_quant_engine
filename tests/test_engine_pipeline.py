from __future__ import annotations

from stock_screener_engine.config.settings import load_settings
from stock_screener_engine.core.engine import ResearchEngine
from stock_screener_engine.data_sources.market.mock_market_data import MockIndianMarketDataProvider
from stock_screener_engine.data_sources.text.mock_text_adapter import MockTextEventProvider


def test_research_engine_outputs() -> None:
    settings = load_settings()
    engine = ResearchEngine(
        settings=settings,
        market_data=MockIndianMarketDataProvider(),
        text_data=MockTextEventProvider(),
    )
    output = engine.run()

    assert output["features"]
    assert output["scores"]
    assert output["long_signals"]
    assert output["swing_signals"]
    assert "long_portfolio_positions" in output
    assert "swing_portfolio_positions" in output
